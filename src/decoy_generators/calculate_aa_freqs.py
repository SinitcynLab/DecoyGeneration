import os
import numpy as np
from random import Random
from typing import List
import torch

from src.io.fasta import write_fasta_file, read_fasta_file

if __name__ == "__main__":
    target_filename: str = "data/targets/UP000002311_559292.fasta"
    records = read_fasta_file(target_filename)

    count_dict = {}
    total_count = 0
    for record in records:
        for aa in record.sequence:
            if aa in count_dict.keys():
                count_dict[aa] += 1
            else:
                count_dict[aa] = 1
            total_count += 1
    
    freq_dict = dict()
    for key in count_dict.keys():
        freq_dict[key] = count_dict[key] / total_count

    print(freq_dict)