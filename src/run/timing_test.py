import os
import numpy as np
from random import Random
from typing import List
import time

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType
from src.decoy_generators.diann_generator import DiannGenerator
from src.decoy_generators.esm_generator import EsmGenerator, MaskingType, MlGeneratorType
from src.decoy_generators.reverse_generator import ReverseGenerator
from src.decoy_generators.shuffle_generator import ShuffleGenerator
from src.decoy_generators.ml_generator import MlGenerator
from src.decoy_generators.random_replace_generator import RandomReplaceGenerator
from src.decoy_generators.smart_masking_esm import MaxProbMaskingEsmGenerator
from src.io.fasta import write_fasta_file, read_fasta_file
from src.io.utils import remove_long_sequences
from src.run.generate import branch_on_generator

def timing_test(test_file: str, N: int, generator_list: List[DecoyGenerator]):
    for generator in generator_list:
        duration = time_generator(test_file, generator, N)
        print(f"{generator}: {duration:.3f} seconds.")

def time_generator(target_file: str, generator: DecoyGenerator, N) -> float:
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
    return end_time - start_time