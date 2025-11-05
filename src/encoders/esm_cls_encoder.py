import torch

from src.encoders.esm_encoder import ESMEncoder
from typing import Iterable

class ESMCLSEncoder(ESMEncoder):
    def __init__(self, device='cpu'):
        ESMEncoder.__init__(self, device=device, constant_length=False)
        self.cls_only = True

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        embeddings = self._embed_batched(sequences) # [n.o. sequences, max_tokenized_length, 320]
        print(embeddings.shape)
        return embeddings