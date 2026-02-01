import random
import numpy as np
import torch

from typing import Iterator, List
from src.io.fasta import FastaRecord

def split_targets(target_sequences: List[str]):
    N = len(target_sequences)//2
    targets = target_sequences[0:N]
    pretend_decoys = target_sequences[N:N*2]

    return targets, pretend_decoys

def remove_long_sequences(records: Iterator[FastaRecord], cap_length: int = 10_000) -> Iterator[FastaRecord]:
    return [rec for rec in records if len(rec.sequence) <= cap_length]

def seed_all(seed):
    # Set the seed for Python's random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    
    # If you are using CUDA (GPU), set the seed for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # For deterministic behavior (e.g., for debugging)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False