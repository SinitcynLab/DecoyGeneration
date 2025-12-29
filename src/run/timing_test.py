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
from src.decoy_generators.diann_modifications import DiannRandomAcid, DiannRandomPos
from src.decoy_generators.ml_generator import MlGenerator
from src.decoy_generators.smart_masking_esm import MaxProbMaskingEsmGenerator, FreqMaskingEsmGenerator, RelDiffMaskingEsmGenerator
from src.decoy_generators.random_replace_generator import RandomReplaceGenerator
from src.io.fasta import write_fasta_file, read_fasta_file
from src.io.utils import remove_long_sequences

if __name__ == "__main__":
    target_filename: str = "data/targets/UP000002311_559292.fasta"
    write_batched: bool = True

    special_amino_acids: List[str] = ['R', 'K']
    device = "cpu" # on CPU because some approaches (e.g. shuffle) don't leverage GPU, making comparison unfair
    # note: you used tue.default.q

    n: int = 1
    N = 100
    random: Random = Random(42)
    generators: List[DecoyGenerator] = [
        DiannGenerator(
            special_amino_acids=special_amino_acids
        ),
        ShuffleGenerator(
            special_amino_acids=special_amino_acids,
            random=random
        ),
        ReverseGenerator(
            special_amino_acids=special_amino_acids
        ),
        RandomReplaceGenerator(
            random=random,
            special_amino_acids=special_amino_acids
        ),
        EsmGenerator(
            local_path="models/esm2_t6_8M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=1
        ),
        EsmGenerator(
            local_path="models/esm2_t33_650M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=1
        ),
        RelDiffMaskingEsmGenerator(
            local_path="models/esm2_t6_8M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device
        )
    ]
    for generator in generators:
        start_time: float = 0
        end_time: float = 0
        filename, extension = os.path.splitext(target_filename)
        if issubclass(type(generator), MlGenerator):
            for i in range(n):
                filename_out = f"{filename}.{generator}.{i}{extension}"
                target_records = [record for record in read_fasta_file(target_filename)]
                target_records = remove_long_sequences(target_records, cap_length=10_000)
                batch_starts = np.arange(0, N, generator.batch_size)
                start_time = time.perf_counter()
                for start in batch_starts:
                    end = min(start + generator.batch_size, len(target_records))
                    write_fasta_file(filename_out, generator.convert_fasta(target_records[start:end]), 60, 'a')
                end_time = time.perf_counter()
        elif generator.decoy_generation_type == DecoyGeneratorType.ONE2ONE:
            filename_out = f"{filename}.{generator}{extension}"
            start_time = time.perf_counter()
            write_fasta_file(filename_out, generator.convert_fasta(read_fasta_file(target_filename)))
            end_time = time.perf_counter()
        elif generator.decoy_generation_type == DecoyGeneratorType.ONE2MANY:
            for i in range(n):
                filename_out = f"{filename}.{generator}.{i}{extension}"
                start_time = time.perf_counter()
                write_fasta_file(filename_out, generator.convert_fasta(read_fasta_file(target_filename)))
                end_time = time.perf_counter()
        print(f"{generator}: {end_time - start_time:.3f} seconds.")
