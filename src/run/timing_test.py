import numpy as np
from typing import List
import time

from src.decoy_generators.decoy_generator import DecoyGenerator
from src.decoy_generators.ml_generator import MlGenerator
from src.io.fasta import read_fasta_file

def timing_test(target_file: str, N: int, generator: DecoyGenerator):
    target_records = [record for record in read_fasta_file(target_file)]
    target_records = target_records[0:N]
    if issubclass(type(generator), MlGenerator):
        batch_starts = np.arange(0, len(target_records), generator.batch_size)
        start_time = time.perf_counter()
        for start in batch_starts:
            end = min(start + generator.batch_size, len(target_records))
            generator.convert_fasta(target_records[start:end])
        end_time = time.perf_counter()
    else:
        start_time = time.perf_counter()
        generator.convert_fasta(target_records)
        end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"{generator}: {duration:.3f} seconds.")
