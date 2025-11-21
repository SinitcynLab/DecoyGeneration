import torch
import copy
import numpy as np
import gc

from typing import Iterable, Union, List, Tuple
from collections.abc import Callable
from src.peptide_classifiers.peptide_classifier import PeptideClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from sklearn.model_selection import StratifiedKFold
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from src.metrics.base_metric import BaseMetric
from src.metrics.default_metric import DefaultMetric
from src.io.data_set import DataSet

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

    def evaluate_on_data(self, dataset: DataSet):
        X = dataset.get_data()
        X.to(self.device)
        y_pred = self(X)
        X.to('cpu')
        return y_pred

    def train_on_data(self, dataset: DataSet, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        raise NotImplementedError()
    
    def reset(self):
        if self.resetter == None:
            raise ValueError("Net setter must be set in order to reset NN classifier.")
        else:
            self.network = self.resetter()
            self.set_device(self.device)

def cross_validate_nn(nn: NNClassifier, main_dataset: DataSet, 
                   n_epochs: int, batch_size: int, learning_grate: float, decoy_id: str, n_folds: int = 5,
                   metric: BaseMetric = DefaultMetric()) -> float:
    print(f"*** *** RESULTS FOR DECOYS={decoy_id} *** ***")
    loss_fn = torch.nn.BCELoss()
    mean_best_val_metrics: np.ndarray = np.zeros(4) # [auc, acc, prec, rec]
    mean_corr_train_metrics: np.ndarray = np.zeros(4)

    kfold = StratifiedKFold(n_splits=n_folds)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(main_dataset.get_data(), main_dataset.get_labels())):
        nn.reset()
        optimizer = torch.optim.Adam(nn.parameters(), lr=learning_grate)

        best_val_metrics: np.ndarray = np.ones(4) * (- np.inf) 
        corr_train_metrics: np.ndarray = np.zeros(4)
        train_dataset: DataSet = main_dataset.get_subset(train_ids)
        val_dataset: DataSet = main_dataset.get_subset(val_ids)

        for _ in range(n_epochs):
            train_metrics, val_metrics = train_val_iteration(nn, train_dataset, val_dataset,
                                                              loss_fn, optimizer, batch_size, metric)
            # use ROC as criterion:
            if val_metrics[0] > best_val_metrics[0]: 
                best_val_metrics = val_metrics
                corr_train_metrics = train_metrics

        print(f"Best validation AUC over #{fold + 1}, with other corresponding metrics:")
        metric.print_values(best_val_metrics)
        print(f"Validation set info on fold #{fold + 1}:")
        print(f"Size: {val_dataset.size()}.")
        print(f"N.o. targets: {val_dataset.get_num_targets()}.")
        print(f"N.o. decoys: {val_dataset.get_num_decoys()}.")

        mean_best_val_metrics += best_val_metrics / n_folds
        mean_corr_train_metrics += corr_train_metrics / n_folds

    print(f"")
    print(f"### Average best validation AUC and corresponding validation metrics over all {n_folds} folds: ###")
    metric.print_values(mean_best_val_metrics)
    print(f"### Average corresponding training metrics over all {n_folds} folds: ###")
    metric.print_values(mean_corr_train_metrics)

    return mean_best_val_metrics[0] # return mean recorded 'best' ROC

def train_nn(nn : NNClassifier, train_dataset: DataSet, val_dataset: DataSet,
            n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    loss_fn = torch.nn.BCELoss()

    best_val_auc = - np.inf
    best_weights = None

    for _ in range(n_epochs):
        _, val_metrics = train_val_iteration(nn, train_dataset, val_dataset, loss_fn, optimizer, batch_size)
        if val_metrics[0] > best_val_auc:
            best_val_auc = val_metrics[0]
            best_weights = copy.deepcopy(nn.state_dict())

    nn.load_state_dict(best_weights) # restore best weights
    return best_val_auc # return best validation auc

def train_val_iteration(nn: NNClassifier, train_dataset: DataSet, val_dataset: DataSet, loss: torch.nn.Module, 
                        optimizer:torch.optim.Optimizer, batch_size: int, metric: BaseMetric = DefaultMetric()):
    # train:
    N: int = train_dataset.size()
    nn.train()
    batch_starts: np.ndarray = np.arange(0, N, batch_size)
    predictions: torch.Tensor = torch.zeros(N)
    for batch_start in batch_starts:
        batch_end: int = min(batch_start + batch_size, N)
        batch_dataset = train_dataset.get_subset(range(batch_start, batch_end))
        y_pred = nn.train_on_data(batch_dataset, loss, optimizer)
        predictions[batch_start:batch_end] = y_pred.cpu()
        del batch_dataset, y_pred
        torch.cuda.empty_cache()
        gc.collect()
    avg_train_metrics = metric.extract_values(predictions, train_dataset.get_labels())

    # validate:
    M: int = val_dataset.size()
    avg_val_metrics: np.ndarray = np.zeros(4)
    nn.eval()
    batch_starts: np.ndarray = np.arange(0, M, batch_size)
    predictions: torch.Tensor = torch.zeros(M)
    for batch_start in batch_starts:
        batch_end: int = min(batch_start + batch_size, M)
        batch_dataset = val_dataset.get_subset(range(batch_start, batch_end))
        y_pred = nn.evaluate_on_data(batch_dataset)
        predictions[batch_start:batch_end] = y_pred.cpu()
        del batch_dataset, y_pred
        torch.cuda.empty_cache()
        gc.collect()
    avg_val_metrics = metric.extract_values(predictions, val_dataset.get_labels())

    return avg_train_metrics, avg_val_metrics