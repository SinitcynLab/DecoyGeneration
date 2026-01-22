import torch
import copy
import numpy as np

from collections.abc import Callable
from src.peptide_classifiers.nn_classifier import NNClassifier, cross_validate_nn, train_nn
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable
from src.metrics.base_metric import BaseMetric
from src.metrics.default_metric import DefaultMetric
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

    def evaluate_on_data(self, tensor_list: Iterable[torch.Tensor], y: torch.Tensor):
        with torch.no_grad():
            X, y = torch.cat(tensor_list, dim=0).to(self.device), y.to(self.device)
            y_pred = self(X)
            del X, y
            torch.cuda.empty_cache()
            return y_pred

    def train_on_data(self, tensor_list: Iterable[torch.Tensor], y: torch.Tensor, 
                    loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        X, y = torch.cat(tensor_list, dim=0).to(self.device), y.to(self.device)
        y_pred = self(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del X, y, loss
        torch.cuda.empty_cache()
        return y_pred