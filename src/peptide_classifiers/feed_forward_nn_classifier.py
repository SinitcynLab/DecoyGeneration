import torch

from collections.abc import Callable
from src.peptide_classifiers.nn_classifier import NNClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable

class FeedForwardNNClassifier(NNClassifier):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder, name: str, device : torch.device, resetter: Callable = None):
        NNClassifier.__init__(self, network, encoder, name, device, resetter)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.network(x))
    
    def set_device(self, device : torch.device):
        NNClassifier.set_device(self, device)

    def evaluate_on_data(self, tensor_list: Iterable[torch.Tensor]):
        with torch.no_grad():
            X = torch.cat(tensor_list, dim=0).to(self.device)
            y_pred = self(X)
            del X
            torch.cuda.empty_cache()
            return y_pred

    def train_on_data(self, tensor_list: Iterable[torch.Tensor], y: torch.Tensor, 
                    loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        X, y = torch.cat(tensor_list, dim=0).to(self.device), y.to(self.device)
        y_pred = self(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del X, y, loss
        torch.cuda.empty_cache()
        return y_pred