import torch

from src.encoders.peptide_encoder import PeptideEncoder
from src.encoders.protbert_encoder import ProtBertEncoder

from typing import Iterable

class ImageEncoder(PeptideEncoder):
    def __init__(self, image_dim : int = 256, device : torch.device = 'cpu'):
        if (image_dim**2) % 1024 != 0:
            raise ValueError("Square of image dimension must be divisible by 1024.")
        max_tokenized_len = image_dim**2 // 1024
        self.image_dim = image_dim
        self.device = device

        self.wrapped_encoder = ProtBertEncoder.__init__(self, max_tokenized_len, device)

    def __call__(self, sequences: Iterable[str]):
        embeddings = self.wrapped_encoder(sequences)
        N = len(sequences)
        return torch.reshape(embeddings, (N, self.image_dim, self.image_dim))