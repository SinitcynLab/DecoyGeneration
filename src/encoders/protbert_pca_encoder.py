import torch

from src.encoders.protbert_encoder import ProtBertEncoder
from src.encoders.pca_encoder import PcaEncoder
from typing import Iterable

class ProtBertPcaEncoder(PcaEncoder):
    def __init__(self, max_tokenized_length : int = 64, device : torch.device = 'cpu', pca_dim : int = 65_536):
        PcaEncoder.__init__(self, ProtBertEncoder(max_tokenized_length, device), pca_dim, device)