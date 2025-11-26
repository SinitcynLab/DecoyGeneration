import lmdb
import pickle
import torch

from torch import Tensor
from typing import Iterable, Tuple

class LMDBDataset(object):
    def __init__(self, lmdb_paths: Iterable[str], labels: Tensor):
        self.envs: Iterable[Tuple[lmdb.Environment, int]] = []
        self.length = 0
        for path in lmdb_paths:
            env = lmdb.open(path, readonly=True, lock=False)
            with env.begin() as txn:
                env_length = txn.stat()['entries']
                self.length += env_length
            self.envs.append((env, env_length))
        if len(labels) != self.length:
            raise ValueError("Ensure that there are as many labels as sequences in the given lmdb directories.")
        self.labels = labels
    
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
        pos_map = {x: i for i, x in enumerate(idx)} # note that the values in idx are all unique
        encodings: Iterable[Tensor] = [torch.zeros(1)] * len(idx)
        labels: Tensor = torch.zeros(len(idx))
        
        # below, the global indices over all lmdb directories are mapped to indices local to each lmdb directory:
        cumulative_size: int = 0
        for (env, env_size) in self.envs:
            env_idx = [i for i in idx if cumulative_size <= i < cumulative_size + env_size]
            with env.begin() as txn:
                for j in env_idx:
                    print(j - cumulative_size)
                    key = f"{j - cumulative_size}".encode() # subtract cumulative size (sum of previous lmdbs) to get local index from global index
                    byte_data = txn.get(key)
                    encodings[pos_map[j]] = pickle.loads(byte_data)
                    labels[pos_map[j]] = self.labels[j]
            cumulative_size += env_size
        return encodings, labels
    
    def get_labels(self, idx: Iterable[int] = None):
        if idx is None:
            idx = range(self.size())        
        return self.labels[idx]