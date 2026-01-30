import torch
import lmdb
import pickle
import numpy as np
import os
import shutil

from typing import Iterable, Tuple
from torch import Tensor

from src.encoders.peptide_encoder import PeptideEncoder

def encode_seqs_to_lmdb(sequences: Iterable[str], encoder: PeptideEncoder, o_file_name: str, batch_size: int = 32):
    if os.path.exists(o_file_name) and os.path.isdir(o_file_name):
        shutil.rmtree(o_file_name)
    os.makedirs(o_file_name)
    batch_starts: np.ndarray = np.arange(0, len(sequences), batch_size)
    for batch_start in batch_starts:
        batch_end = min(len(sequences), batch_start + batch_size)
        batch_encodings = encoder(sequences[batch_start:batch_end])
        if isinstance(batch_encodings, torch.Tensor):
            batch_encodings = list(batch_encodings.split(1, dim=0)) # if single tensor returned, unroll into a list of tensors
        append_tensors_to_lmdb(batch_encodings, range(batch_start, batch_end), o_file_name)

def append_tensors_to_lmdb(tensors: Iterable[torch.Tensor], indices: Iterable[int], out_file: str):
    env: lmdb.Environment = lmdb.open(out_file, map_size=1024**4)
    pairs: Iterable[Tuple[int, Tensor]] = zip(indices, tensors) # note that this will iterate over list of tensors/first dim of tensor containing batch
    with env.begin(write=True) as txn:
        for (i, t) in pairs:
            key: str = f"{i}".encode() # label tensor by index
            txn.put(key, pickle.dumps(t)) # store tensor in lmdb with index as key

def delete_lmdb(path: str):
    for name in os.listdir(path):
        full: str = os.path.join(path, name)
        if os.path.isfile(full) or os.path.islink(full):
            os.remove(full)
        elif os.path.isdir(full):
            shutil.rmtree(full)
    if os.path.isdir(path) and not os.listdir(path):  # empty directory
                os.rmdir(path)