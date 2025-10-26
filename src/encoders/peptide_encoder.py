import torch
from typing import Iterable

class PeptideEncoder(object):
    def __init__(self):
        object.__init__(self)

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        raise NotImplementedError()
    
    def normalize_tensor(self, x : torch.Tensor) -> torch.Tensor:
        for i in range(x.shape[1]):
            min = x[:,i].min()
            max = x[:,i].max()
            if (max - min) > 0:
                x[:,i] = (x[:,i] - min) / (max - min)
        return x