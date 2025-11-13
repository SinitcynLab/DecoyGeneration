import os
import numpy as np
from random import Random
from typing import List
import torch

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType
from src.decoy_generators.diann_generator import DiannGenerator
from src.decoy_generators.esm_generator import EsmGenerator, MlGeneratorType
from src.decoy_generators.reverse_generator import ReverseGenerator
from src.decoy_generators.shuffle_generator import ShuffleGenerator
from src.io.fasta import write_fasta_file, read_fasta_file

if __name__ == "__main__":
    target_filename: str = "data/targets/UP000002311_559292.fasta"
    write_batched: bool = True

    special_amino_acids: List[str] = ['R', 'K']
    torch.set_num_threads(6)

    n: int = 3
    random: Random = Random(42)
    generators: List[DecoyGenerator] = [
        EsmGenerator(
            local_path="models/esm2_t36_3B_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            mask_percent=0.3,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST
        )
    ]
    for generator in generators:
        filename, extension = os.path.splitext(target_filename)
        if type(generator) in [EsmGenerator]:
            for i in range(n):
                filename_out = f"{filename}.{generator}.{i}{extension}"
                target_records = [record for record in read_fasta_file(target_filename)]
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
