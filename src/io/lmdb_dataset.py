import lmdb
import pickle
import torch

from torch import Tensor
from typing import Iterable, Tuple
from src.encoders.transformer_encoder import pad_tensor_list

class LMDBDataset(object):
    def __init__(self, lmdb_paths: Iterable[str], labels: Tensor):
        self.envs: Iterable[Tuple[lmdb.Environment, int]] = []
        for path in lmdb_paths:
            env = lmdb.open(path, readonly=True, lock=False)
            with env.begin() as txn:
                env_length = txn.stat()['entries']
                self.length += env_length
            self.envs.append[(env, env_length)]
        if len(labels) != self.length:
            raise ValueError("Ensure that there are as many labels as sequences in the given lmdb directories.")
        self.labels = labels
    
    def size(self):
        return self.length
    
    def get_num_targets(self):
        return (self.labels == 0.).sum(dim=0)
    
    def get_num_decoys(self):
        return (self.labels == 1.).sum(dim=0)

    def _get_sample_list(self, idx: Iterable[int]):
        pos_map = {x: i for i, x in enumerate(idx)} # note that the values in idx are all unique
        encodings: Iterable[Tensor] = [torch.zeros(1)] * len(idx)
        labels: Tensor = torch.zeros(len(idx))
        
        cumulative_size: int = 0
        for (env, env_size) in self.envs:
            env_idx = [i for i in idx if i < cumulative_size + env_size]
            with env.begin() as txn:
                for j in env_idx:
                    key = f"{j}".encode()
                    byte_data = txn.get(key)
                    encodings[pos_map[j]] = pickle.loads(byte_data)
                    labels[pos_map[j]] = self.labels[j]
            cumulative_size += env_size
        return encodings, labels
    
    def get_samples(self, idx: Iterable[int]):
        encodings, labels = self._get_sample_list(idx)
        return torch.cat(encodings, dim = 0), labels
    
class LMDBDataset(LMDBDataset):
    def __init__(self, lmdb_paths, labels):
        LMDBDataset.__init__(self, lmdb_paths, labels)

    def get_samples(self, idx):
        encodings, labels = self._get_sample_list(idx)
        X, l = pad_tensor_list(encodings)
        return X, l, labels
