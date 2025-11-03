import torch
import copy
import numpy as np

from src.peptide_classifiers.nn_classifier import NNClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable

class FeedForwardNNClassifier(NNClassifier, torch.nn.Module):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder, device : torch.device):
        NNClassifier.__init__(self, network, encoder, device)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.network(x))
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        x : torch.Tensor = self.encoder(sequences)
        x = self.forward(x)
        res = torch.round(x).tolist()
        return res
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> tuple[list[bool], list[float]]:
        x : torch.Tensor = self.encoder(sequences)
        x = self.forward(x)
        pred : torch.Tensor = torch.round(x)
        outcomes_tensor = torch.IntTensor(list(outcomes)).to(self.device)
        corr : torch.Tensor = torch.eq(pred, outcomes_tensor)
        print(x.device)
        loss : torch.Tensor = torch.abs(x - outcomes_tensor.float())
        return corr.tolist(), loss.tolist()
    
    def set_device(self, device : torch.device):
        NNClassifier.set_device(device)