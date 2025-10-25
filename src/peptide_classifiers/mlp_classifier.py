import torch
import math

from src.peptide_classifiers.peptide_classifier import PeptideClassifier
from typing import Iterable

class mlp_classifier(PeptideClassifier, torch.nn.Module):
    def __init__(self, network:torch.nn.Sequential, encoder):
        PeptideClassifier.__init__(self, encoder)
        torch.nn.Module.__init__(self)
        self.network = network

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        x : torch.Tensor = self.encoder(sequences)
        x = self.forward(x)
        res = torch.round(x).tolist()
        return res
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> list[tuple[bool, float]]:
        x : torch.Tensor = self.encoder(sequences)
        x = self.forward(x)
        pred : torch.Tensor = torch.round(x)
        corr : torch.Tensor = (pred == outcomes)
        outcomes_tensor = torch.FloatTensor(list(outcomes))
        loss : torch.Tensor = torch.abs(x - outcomes_tensor)
        return list(zip(corr.tolist(), loss.tolist()))

