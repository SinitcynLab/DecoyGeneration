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
    
    def set_device(self, device : torch.device):
        NNClassifier.set_device(device)