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
    special_amino_acids: List[str] = ['R', 'K']
    device = "cpu" # on CPU because some approaches (e.g. shuffle) don't leverage GPU, making comparison unfair
    n: int = 1

    for generator in generator_list:
        start_time: float = 0
        end_time: float = 0
        branch_on_generator(test_file, 1, generator)
        print(f"{generator}: {end_time - start_time:.3f} seconds.")