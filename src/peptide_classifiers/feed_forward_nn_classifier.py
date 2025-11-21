import torch
import copy
import numpy as np

from collections.abc import Callable
from src.peptide_classifiers.nn_classifier import NNClassifier, cross_validate_nn, train_nn
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable
from src.metrics.base_metric import BaseMetric
from src.metrics.default_metric import DefaultMetric
from src.io.data_set import DataSet
from sklearn.utils import shuffle

class FeedForwardNNClassifier(NNClassifier):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder, name: str, device : torch.device, resetter: Callable = None):
        NNClassifier.__init__(self, network, encoder, name, device, resetter)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.network(x))
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        # get inputs:
        encodings: torch.Tensor = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings)
        # cast outputs to list and return:
        return torch.round(out).tolist()
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> tuple[list[bool], list[float]]:
        # get inputs:
        encodings: torch.Tensor = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings)
        # compute predictions, get accuracy:
        pred : torch.Tensor = torch.round(out)
        outcomes_tensor = torch.IntTensor(list(outcomes)).to(self.device)
        corr : torch.Tensor = torch.eq(pred, outcomes_tensor)
        loss : torch.Tensor = torch.abs(out - outcomes_tensor.float())
        # return whether predictions correct, and loss measure with each prediction:
        return corr.tolist(), loss.tolist()
    
    def set_device(self, device : torch.device):
        NNClassifier.set_device(self, device)

    def evaluate_on_data(self, dataset: DataSet):
        dataset.to(self.device)
        X, _ = dataset.get_tensors()
        y_pred = self(X)
        dataset.to('cpu')
        return y_pred

    def train_on_data(self, dataset: DataSet, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        dataset.to(self.device)
        X, y = dataset.get_tensors()
        y_pred = self(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        dataset.to('cpu')
        return y_pred

def cross_validate_ff_nn(ff_nn: FeedForwardNNClassifier, sequences: Iterable[str], labels: Iterable[str], 
                   n_epochs: int, batch_size: int, learning_rate: float, decoy_id: str, n_folds: int = 5,
                   metric: BaseMetric = DefaultMetric()) -> float:
    sequences, labels = shuffle(sequences, labels)
    data: torch.Tensor = ff_nn.encoder(sequences)
    labels: torch.Tensor = torch.FloatTensor(list(labels))
    dataset: DataSet = DataSet(data, labels)
    return cross_validate_nn(ff_nn, dataset, n_epochs, batch_size, learning_rate, decoy_id, n_folds, metric)

def train_ff_nn(ff_nn : FeedForwardNNClassifier, X_train : Iterable[str], y_train : Iterable[bool], X_val : Iterable[str], 
              y_val : Iterable[str], n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    X_train = ff_nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train))
    X_val = ff_nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val))
    
    train_dataset = DataSet(X_train, y_train)
    val_dataset = DataSet(X_val, y_val)

    return train_nn(ff_nn, train_dataset, val_dataset, n_epochs, batch_size, optimizer)