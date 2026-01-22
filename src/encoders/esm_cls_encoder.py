import torch

from src.encoders.esm_encoder import EsmEncoder
from typing import Iterable

class ESMCLSEncoder(EsmEncoder):
    def __init__(self, device='cpu'):
        EsmEncoder.__init__(self, device=device, constant_length=False)
        self.cls_only = True

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        embeddings = self._embed_batched_constant_length(sequences) # [n.o. sequences, max_tokenized_length, 320]
        return embeddings