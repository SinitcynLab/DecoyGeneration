import os
import numpy as np
from random import Random
from typing import List
import torch

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType
from src.decoy_generators.diann_generator import DiannGenerator
from src.decoy_generators.esm_generator import EsmGenerator, MaskingType, MlGeneratorType
from src.decoy_generators.reverse_generator import ReverseGenerator
from src.decoy_generators.shuffle_generator import ShuffleGenerator
from src.decoy_generators.diann_modifications import DiannRandomAcid, DiannRandomPos
from src.decoy_generators.ml_generator import MlGenerator
from src.decoy_generators.smart_masking_esm import MaxProbMaskingEsmGenerator, FreqMaskingEsmGenerator, RelDiffMaskingEsmGenerator, SimMaskingEsmGenerator
from src.decoy_generators.random_replace_generator import RandomReplaceGenerator
from src.io.fasta import write_fasta_file, read_fasta_file
from src.io.utils import remove_long_sequences

if __name__ == "__main__":
    target_filename: str = "data/targets/UP000002311_559292.fasta"
    write_batched: bool = True

    special_amino_acids: List[str] = ['R', 'K']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    n: int = 1
    random: Random = Random(42)
    generators: List[DecoyGenerator] = [
        RandomReplaceGenerator(
            special_amino_acids=special_amino_acids,
            random=random
        )
    ]
    for generator in generators:
        filename, extension = os.path.splitext(target_filename)
        if issubclass(type(generator), MlGenerator):
            for i in range(n):
                filename_out = f"{filename}.{generator}.{i}{extension}"
                target_records = [record for record in read_fasta_file(target_filename)]
                target_records = remove_long_sequences(target_records, cap_length=10_000)
                batch_starts = np.arange(0, len(target_records), generator.batch_size)
                for start in batch_starts:
                    end = min(start + generator.batch_size, len(target_records))
                    write_fasta_file(filename_out, generator.convert_fasta(target_records[start:end]), 60, 'a')
                    print(f"{end}/{len(target_records)}")
        elif generator.decoy_generation_type == DecoyGeneratorType.ONE2ONE:
            filename_out = f"{filename}.{generator}{extension}"
            write_fasta_file(filename_out, generator.convert_fasta(read_fasta_file(target_filename)))
        elif generator.decoy_generation_type == DecoyGeneratorType.ONE2MANY:
            for i in range(n):
                filename_out = f"{filename}.{generator}.{i}{extension}"
                write_fasta_file(filename_out, generator.convert_fasta(read_fasta_file(target_filename)))
