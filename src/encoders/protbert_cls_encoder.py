import torch

from src.encoders.protbert_encoder import ProtBertEncoder
from typing import Iterable

class ProtBertClsEncoder(ProtBertEncoder):
    def __init__(self, device='cpu'):
        ProtBertEncoder.__init__(self, device=device, constant_length=False)
        self.cls_only = True

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        sequences = [" ".join(sequence) for sequence in sequences]
        embeddings = self._embed_batched_varied_length(sequences) # [Batch, 1, 1024]
        return embeddings