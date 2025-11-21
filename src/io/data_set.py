import torch

from typing import Iterable
from torch import Tensor

class DataSet(object):
    def __init__(self, data: Tensor, labels: Tensor):
        self.data = data
        self.labels = labels

    def to(self, device: torch.device):
        self.data.to(device)
        self.labels.to(device)
    
    def get_tensors(self):
        return self.data, self.labels
    
    def get_labels(self):
        return self.labels
    
    def get_data(self):
        return self.data

    def get_subset(self, idx: Iterable[int]):
        return DataSet(self.data[idx], self.labels[idx])
    
    def size(self):
        return len(self.labels)
    
    def get_num_targets(self):
        return (self.labels == 0.).sum(dim=0)
    
    def get_num_decoys(self):
        return (self.labels == 1.).sum(dim=0)
    
class RecurrentDataSet(DataSet):
    def __init__(self, data: Tensor, labels: Tensor, lengths: Tensor):
        DataSet.__init__(self, data, labels)
        self.lengths = lengths

    def to(self, device: torch.device):
        DataSet.to(self, device)
        self.lengths.to(device)

    def get_tensors(self):
        return self.data, self.labels, self.lengths

    def get_subset(self, idx: Iterable[int]):
        return RecurrentDataSet(self.data[idx], self.labels[idx], self.lengths[idx])
    
    def get_lengths(self):
        return self.lengths