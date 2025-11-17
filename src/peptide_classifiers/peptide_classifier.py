import torch
from typing import Iterable

from src.encoders.peptide_encoder import PeptideEncoder

class PeptideClassifier(object):
    def __init__(self, encoder : PeptideEncoder, name: str, device : torch.device = 'cpu'):
        object.__init__(self)
        self.encoder = encoder
        self.device=device
        self.name = name
    
    # take sequence and return predicted class (0 = real, 1 = decoy)
    def classify(sequences: Iterable[str]) -> list[bool]:
        raise NotImplementedError()

    # take sequence and class (0 = real, 1 = decoy), return whether correct classification and loss measure
    def score(sequences: Iterable[str], outcomes: Iterable[bool]) -> list[tuple[bool, float]]:
        raise NotImplementedError()
    
    def set_device(self, device: torch.device):
        self.device = device

    def __str__(self):
        return self.name