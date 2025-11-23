import torch

from typing import Iterable
from torch import Tensor

class Dataset(object):
    def __init__(self, sequences: Iterable[str], labels: Tensor):
        self.sequences = sequences
        self.labels = labels

    def get_contents(self):
        return self.sequences, self.labels
    
    def get_labels(self):
        return self.labels
    
    def get_sequences(self):
        return self.sequences

    def get_subset(self, idx: Iterable[int]):
        return Dataset(self.sequences[idx], self.labels[idx])
    
    def size(self):
        return len(self.labels)
    
    def get_num_targets(self):
        return (self.labels == 0.).sum(dim=0)
    
    def get_num_decoys(self):
        return (self.labels == 1.).sum(dim=0)