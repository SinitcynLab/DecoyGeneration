import torch

from src.encoders.peptide_encoder import PeptideEncoder
from src.encoders.protbert_encoder import ProtBertEncoder

from typing import Iterable

class ImageEncoder(PeptideEncoder):
    def __init__(self, image_height : int = 256, device : torch.device = 'cpu'):
        PeptideEncoder.__init__(self)
        self.image_height = image_height
        self.device = device

        self.wrapped_encoder = ProtBertEncoder(image_height, device)

    def __call__(self, sequences: Iterable[str]):
        embeddings = self.wrapped_encoder(sequences)
        N = len(sequences)
        return torch.reshape(embeddings, (N, 1, 1024, self.image_height)) # [N, C, W, H]