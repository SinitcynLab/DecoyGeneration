import torch
from typing import Iterable

from src.encoders.peptide_encoder import PeptideEncoder

class PeptideClassifier(object):
    def __init__(self, encoder : PeptideEncoder, name: str, device : torch.device = 'cpu'):
        object.__init__(self)
        self.encoder = encoder
        self.device = device
        self.name = name
    
    def set_device(self, device: torch.device):
        self.device = device

    def __str__(self):
        return self.name