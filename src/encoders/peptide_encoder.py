import torch
from typing import Iterable

class PeptideEncoder(object):
    def __init__(self):
        object.__init__(self)
        self.canonical_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        raise NotImplementedError()