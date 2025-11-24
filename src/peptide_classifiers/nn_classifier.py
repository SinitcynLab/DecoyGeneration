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
from src.io.data_set import Dataset
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

    def evaluate_on_data(self, dataset: Dataset):
        return NotImplementedError()

    def train_on_data(self, dataset: Dataset, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        raise NotImplementedError()
    
    def reset(self):
        if self.resetter == None:
            raise ValueError("Net setter must be set in order to reset NN classifier.")
        else:
            self.network = self.resetter()
            self.set_device(self.device)

    def encode_dataset(self, dataset: Dataset):
        raise NotImplementedError()

    def gc_tensors(self, tensor_list: List[torch.Tensor]):
        for t in tensor_list:
            del t
        gc.collect()
        torch.cuda.empty_cache()

    def gc_tensors(self, tensor_list: Iterable[torch.Tensor]):
        for t in tensor_list:
            del t
        torch.cuda.empty_cache()
        gc.collect()

def cross_validate_nn(nn: NNClassifier, sequences: Iterable[str], labels: Iterable[bool], 
                   n_epochs: int, batch_size: int, learning_grate: float, decoy_id: str, n_folds: int = 5,
                   metric: BaseMetric = DefaultMetric()) -> float:
    print(f"*** *** RESULTS FOR DECOYS={decoy_id} *** ***")
    sequences = np.array(sequences, dtype=str) # convert to array for indexing folds
    sequences, labels = shuffle(sequences, labels)
    labels: torch.Tensor = torch.FloatTensor(list(labels))
    main_dataset: Dataset = Dataset(sequences, labels)

    loss_fn = torch.nn.BCELoss()
    best_val_metrics: np.ndarray = np.zeros((n_folds, metric.dim)) # [auc, acc, prec, rec], one for each fold
    corr_train_metrics: np.ndarray = np.zeros((n_folds, metric.dim))

    kfold = StratifiedKFold(n_splits=n_folds)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(main_dataset.get_sequences(), main_dataset.get_labels())):
        nn.reset()
        optimizer = torch.optim.Adam(nn.parameters(), lr=learning_grate)

        best_val_fold: np.ndarray = np.ones(4) * (- np.inf) 
        corr_train_fold: np.ndarray = np.zeros(4)
        train_dataset: Dataset = main_dataset.get_subset(train_ids)
        val_dataset: Dataset = main_dataset.get_subset(val_ids)

        for _ in range(n_epochs):
            train_metrics, val_metrics = train_val_iteration(nn, train_dataset, val_dataset,
                                                              loss_fn, optimizer, batch_size, metric)
            # use ROC as criterion:
            if val_metrics[0] > best_val_fold[0]: 
                best_val_fold = val_metrics
                corr_train_fold = train_metrics

        print(f"Best validation AUC over #{fold + 1}, with other corresponding metrics:")
        metric.print_metric(best_val_fold)
        print(f"Validation set info on fold #{fold + 1}:")
        print(f"Size: {val_dataset.size()}.")
        print(f"N.o. targets: {val_dataset.get_num_targets()}.")
        print(f"N.o. decoys: {val_dataset.get_num_decoys()}.")

        best_val_metrics[fold,:] = best_val_fold
        corr_train_metrics[fold,:] = corr_train_fold

    print(f"### Average best validation AUC and corresponding validation metrics over all {n_folds} folds: ###")
    metric.print_metric_series(best_val_metrics)
    print(f"### Average corresponding training metrics over all {n_folds} folds: ###")
    metric.print_metric_series(corr_train_metrics)

    print(f"")
    return best_val_metrics[0] # return mean recorded 'best' ROC

def train_nn(nn : NNClassifier, X_train : Iterable[str], y_train : Iterable[bool], X_val : Iterable[str], 
            y_val : Iterable[str], n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    y_train = torch.FloatTensor(list(y_train))
    y_val = torch.FloatTensor(list(y_val))
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)

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

def train_val_iteration(nn: NNClassifier, train_dataset: Dataset, val_dataset: Dataset, loss: torch.nn.Module, 
                        optimizer:torch.optim.Optimizer, batch_size: int, metric: BaseMetric = DefaultMetric()):
    # train:
    N: int = train_dataset.size()
    nn.train()
    batch_starts: np.ndarray = np.arange(0, N, batch_size)
    predictions: torch.Tensor = torch.zeros(N)
    for batch_start in batch_starts:
        batch_end: int = min(batch_start + batch_size, N)
        print(f"{batch_end}/{N}")
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