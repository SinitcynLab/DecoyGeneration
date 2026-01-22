import lmdb
import pickle
import torch

from torch import Tensor
from typing import Iterable, Tuple, List
from src.encoders.peptide_encoder import PeptideEncoder

class TensorDataset(object):
    def __init__(self, sequences: List[str], labels: Tensor):
        self.sequences = sequences
        self.length = len(labels)
        self.labels = labels

    def encode(self, encoder: PeptideEncoder, batch_size: int):
        batch_starts = torch.arange(0, self.length, batch_size)
        self.encodings = []
        for batch_start in batch_starts:
            batch_end = min(self.length, batch_start + batch_size)
            encodings_tensor = encoder(self.sequences[batch_start:batch_end])
            self.encodings = self.encodings + list(encodings_tensor.split(dim=0, split_size=1))
    
    def size(self):
        return self.length
    
    def get_num_targets(self, idx: Iterable[int] = None):
        if idx is None:
            idx = range(self.size())
        return (self.labels[idx] == 0.).sum(dim=0)
    
    def get_num_decoys(self, idx: Iterable[int]):
        if idx is None:
            idx = range(self.size())
        return (self.labels[idx] == 1.).sum(dim=0)

    # if you e.g. feed it indices [123, 203, 24]
    # it would return [tensor_corr_to_123, tensor_corr_to_203, tensor_corr_to_24], [label_of_123, label_of_203, label_of_24]
    def get_pairs(self, idx: Iterable[int]):
        pair_encodings = [self.encodings[i] for i in idx]
        pair_labels = self.labels[idx]
        return pair_encodings, pair_labels
    
    def get_labels(self, idx: Iterable[int] = None):
        if idx is None:
            idx = range(self.size())        
        return self.labels[idx]