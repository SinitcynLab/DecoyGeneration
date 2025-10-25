import torch
from typing import Iterable

class PeptideEncoder(object):
    def __init__(self):
        object.__init__(self)

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        raise NotImplementedError()