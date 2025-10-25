import torch
from typing import Iterable

class PeptideClassifier(object):
    def __init__(self, encoder):
        object.__init__(self)
        self.encoder = encoder
    
    # take sequence and return predicted class (0 = real, 1 = decoy)
    def classify(sequences: Iterable[str]) -> list[bool]:
        raise NotImplementedError()

    # take sequence and class (0 = real, 1 = decoy), return whether correct classification and loss measure
    def score(sequences: Iterable[str], outcomes: Iterable[bool]) -> list[tuple[bool, float]]:
        raise NotImplementedError()