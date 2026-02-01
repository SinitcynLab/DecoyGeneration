import torch
import copy
import numpy as np

from typing import Iterable
from collections.abc import Callable
from src.peptide_classifiers.peptide_classifier import PeptideClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from sklearn.model_selection import StratifiedKFold
from src.metrics.base_metric import BaseMetric
from src.metrics.default_metric import DefaultMetric
from src.io.lmdb_dataset import LMDBDataset
from sklearn.utils import shuffle

class NNClassifier(PeptideClassifier, torch.nn.Module):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder, name: str, device : torch.device, resetter: Callable = None):
        PeptideClassifier.__init__(self, encoder, name, device)
        torch.nn.Module.__init__(self)
        self.network = network
        self.network.to(device)
        self.resetter = resetter

    def set_device(self, device : torch.device):
        PeptideClassifier.set_device(self, device)
        self.network.to(device)

    def evaluate_on_data(self, dataset: LMDBDataset):
        return NotImplementedError()

    def train_on_data(self, dataset: LMDBDataset, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        raise NotImplementedError()
    
    def reset(self):
        if self.resetter == None:
            raise ValueError("Net setter must be set in order to reset NN classifier.")
        else:
            self.network = self.resetter()
            self.set_device(self.device)

def cross_validate_nn(nn: NNClassifier, main_dataset: LMDBDataset, 
                   n_epochs: int, batch_size: int, learning_rate: float, decoy_id: str, weight_decay: float = 0, n_folds: int = 5,
                   metric: BaseMetric = DefaultMetric()) -> float:
    print(f"*** *** RESULTS FOR DECOYS={decoy_id} *** ***")
    N = main_dataset.size()

    loss_fn = torch.nn.BCELoss()
    best_val_metrics: np.ndarray = np.zeros((n_folds, metric.dim)) # [auc, acc, prec, rec], one for each fold
    corr_train_metrics: np.ndarray = np.zeros((n_folds, metric.dim))

    kfold = StratifiedKFold(n_splits=n_folds)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(np.zeros(N), main_dataset.get_labels())):
        train_ids: np.ndarray = shuffle(train_ids)
        val_ids: np.ndarray = shuffle(val_ids)
        
        nn.reset()
        optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate)

        best_val_fold: np.ndarray = np.ones(4) * (- np.inf) 
        corr_train_fold: np.ndarray = np.zeros(4)

        for epoch in range(n_epochs):
            train_metrics, val_metrics = train_val_iteration(nn, main_dataset, train_ids, val_ids,
                                                              loss_fn, optimizer, batch_size, epoch, metric)
            print(f"epoch: {epoch+1}/{n_epochs}")
            # use ROC as criterion:
            if val_metrics[0] > best_val_fold[0]: 
                best_val_fold = val_metrics
                corr_train_fold = train_metrics

        print(f"Best validation AUC over #{fold + 1}, with other corresponding metrics:")
        metric.print_metric(best_val_fold)
        print(f"Validation set info on fold #{fold + 1}:")
        print(f"Size: {len(val_ids)}.")
        print(f"N.o. targets: {main_dataset.get_num_targets(val_ids)}.")
        print(f"N.o. decoys: {main_dataset.get_num_decoys(val_ids)}.")

        best_val_metrics[fold,:] = best_val_fold
        corr_train_metrics[fold,:] = corr_train_fold
        print(f"{fold}/{n_folds}")

    print(f"### Average best validation AUC and corresponding validation metrics over all {n_folds} folds: ###")
    metric.print_metric_series(best_val_metrics)
    print(f"### Average corresponding training metrics over all {n_folds} folds: ###")
    metric.print_metric_series(corr_train_metrics)

    print(f"")
    return np.mean(best_val_metrics[:,0]) # return mean recorded 'best' ROC

def train_nn(nn : NNClassifier, dataset: LMDBDataset, train_ids: Iterable[int], val_ids: Iterable[int],
            n_epochs : int, batch_size : int, learning_rate : float, weight_decay: float = 0, metric: BaseMetric = DefaultMetric()):
    optimizer = torch.optim.Adam(params=nn.parameters(), lr = learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCELoss()

    best_val_auc = - np.inf
    best_weights = None

    for epoch in range(n_epochs):
        _, val_metrics = train_val_iteration(nn, dataset, train_ids, val_ids, loss_fn, optimizer, batch_size, epoch, metric)
        if val_metrics[0] > best_val_auc:
            best_val_auc = val_metrics[0]
            best_weights = copy.deepcopy(nn.state_dict())

    nn.load_state_dict(best_weights) # restore best weights
    metric.print_metric(val_metrics)
    return best_val_auc # return best validation auc

def train_val_iteration(nn: NNClassifier, dataset: LMDBDataset, train_ids: Iterable[int], val_ids: Iterable[int], loss_fn: torch.nn.Module, 
                        optimizer:torch.optim.Optimizer, batch_size: int, epoch: int = 0, metric: BaseMetric = DefaultMetric()):
    # train:
    N: int = len(train_ids)
    nn.train()
    batch_starts: np.ndarray = np.arange(0, N, batch_size) # get the starts of the batches
    predictions: torch.Tensor = torch.zeros(N) # create a tensor to hold the predictions
    for batch_start in batch_starts:
        batch_end: int = min(batch_start + batch_size, N)
        batch_ids = train_ids[batch_start:batch_end] # get the indices of training data
        t_list, y = dataset.get_pairs(batch_ids) # get tensors and targets for training
        y_pred = nn.train_on_data(t_list, y, loss_fn, optimizer) # train neural net on data
        predictions[batch_start:batch_end] = y_pred.cpu() # save predictions in premade tensor
        del y_pred, t_list, y
        torch.cuda.empty_cache()
    avg_train_metrics = metric.extract_values(predictions, dataset.get_labels(train_ids)) # use predictions and labels to evaluate

    # validate:
    M: int = len(val_ids)
    avg_val_metrics: np.ndarray = np.zeros(4)
    nn.eval()
    batch_starts: np.ndarray = np.arange(0, M, batch_size) # get the starts of the batches
    predictions: torch.Tensor = torch.zeros(M) # create a tensor to hold the predictions
    for batch_start in batch_starts:
        batch_end: int = min(batch_start + batch_size, M)        
        batch_ids = val_ids[batch_start:batch_end] # get the indices of evaluation data
        t_list, _ = dataset.get_pairs(batch_ids) # get the tensors and targets for evaluation
        y_pred = nn.evaluate_on_data(t_list) # evaluate neural net on data
        predictions[batch_start:batch_end] = y_pred.cpu() # save predictions in premade tensor
        del y_pred, t_list
        torch.cuda.empty_cache()
    avg_val_metrics = metric.extract_values(predictions, dataset.get_labels(val_ids)) # use predictions and labels to evaluate

    return avg_train_metrics, avg_val_metrics