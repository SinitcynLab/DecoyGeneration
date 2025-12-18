import os
import numpy as np
from random import Random
from typing import List
import torch

from src.io.fasta import write_fasta_file, read_fasta_file

if __name__ == "__main__":
    target_filename: str = "data/targets/UP000002311_559292.fasta.mass_smart_masking_esm_8M.0.fasta"
    fasta_records = read_fasta_file(target_filename)
    fasta_records = [record for record in fasta_records]
    fasta_records = fasta_records[0:3008]
    write_fasta_file("data/decoys/UP000002311_559292.mass_smart_masking_esm_8M.0.fasta", fasta_records)