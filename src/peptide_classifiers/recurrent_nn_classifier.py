import torch
import copy
import numpy as np

from collections.abc import Callable
from typing import Iterable, Tuple
from src.peptide_classifiers.nn_classifier import NNClassifier, cross_validate_nn, train_nn
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable
from src.metrics.base_metric import BaseMetric
from src.metrics.default_metric import DefaultMetric
from sklearn.utils import shuffle
from src.io.data_set import RecurrentDataSet

class RecurrentNNClassifier(NNClassifier):
    def __init__(self, rnn : torch.nn.RNN, network : torch.nn.Sequential, encoder : PeptideEncoder, name: str, device : torch.device, resetter: Callable = None):
        NNClassifier.__init__(self, network, encoder, name, device, resetter)
        self.rnn = rnn
        self.rnn.to(self.device)

    def forward(self, data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=True)
        rnn_out, _ = self.rnn(packed_seqs)
        output = self.network(rnn_out[:, -1, :]) # take output at final step
        return output
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        # get inputs:
        encodings, lengths = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings, lengths)
        # cast outputs to list and return:
        return torch.round(out).tolist()
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> tuple[list[bool], list[float]]:
        # get inputs:
        encodings, lengths = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings, lengths)
        # compute predictions, get accuracy:
        pred : torch.Tensor = torch.round(out)
        outcomes_tensor = torch.IntTensor(list(outcomes)).to(self.device)
        corr : torch.Tensor = torch.eq(pred, outcomes_tensor)
        loss : torch.Tensor = torch.abs(out - outcomes_tensor.float())
        # return whether predictions correct, and loss measure with each prediction:
        return corr.tolist(), loss.tolist()
    
    def train_on_data(self, dataset: RecurrentDataSet, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        dataset.to(self.device)
        X, y, l = dataset.get_tensors()
        y_pred = self(X, l)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        dataset.to('cpu')
        return y_pred
    
    def set_device(self, device):
        NNClassifier.set_device(self, device)
        self.rnn.to(device)
    
    def reset(self):
        net, rnn = self.resetter()
        self.network = net
        self.rnn = rnn
        self.set_device(self.device)

def cross_validate_rnn(rnn: RecurrentNNClassifier, sequences: Iterable[str], labels: Iterable[str], 
                   n_epochs: int, batch_size: int, learning_rate: float, decoy_id: str, n_folds: int = 5,
                   metric: BaseMetric = DefaultMetric()) -> float:
    sequences, labels = shuffle(sequences, labels)
    data, lengths = rnn.encoder(sequences)
    labels: torch.Tensor = torch.FloatTensor(list(labels))
    dataset: RecurrentDataSet = RecurrentDataSet(data, labels, lengths)
    return cross_validate_nn(rnn, dataset, n_epochs, batch_size, learning_rate, decoy_id, n_folds, metric)

def train_rnn(rnn : RecurrentNNClassifier, X_train : Iterable[str], y_train : Iterable[bool], X_val : Iterable[str], 
              y_val : Iterable[str], n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    X_train, len_train = rnn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train))
    X_val, len_val = rnn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val))
    
    train_dataset = RecurrentDataSet(X_train, y_train, len_train)
    val_dataset = RecurrentDataSet(X_val, y_val, len_val)

    return train_nn(rnn, train_dataset, val_dataset, n_epochs, batch_size, optimizer)