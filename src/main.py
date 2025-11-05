import os
from random import Random
from typing import List

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType
from src.decoy_generators.diann_generator import DiannGenerator
from src.decoy_generators.esm_generator import EsmGenerator, EsmGeneratorType
from src.decoy_generators.reverse_generator import ReverseGenerator
from src.decoy_generators.shuffle_generator import ShuffleGenerator
from src.io.fasta import write_fasta_file, read_fasta_file

if __name__ == "__main__":
    target_filename: str = "data/targets/UP000002311_559292.fasta"

    special_amino_acids: List[str] = ['R', 'K']

    n: int = 3
    random: Random = Random(42)
    generators: List[DecoyGenerator] = [
        EsmGenerator(
            local_path="models/esm2_t6_8M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            mask_percent=0.3,
            sort_optimization=True,
            batch_size=64,
            esm_generator_type=EsmGeneratorType.BEST
        )
    ]
    for generator in generators:
        filename, extension = os.path.splitext(target_filename)
        switch: object
        match generator.decoy_generation_type:
            case DecoyGeneratorType.ONE2ONE:
                filename_out = f"{filename}.{generator}{extension}"
                write_fasta_file(filename_out, generator.convert_fasta(read_fasta_file(target_filename)[1:1000]))
            case DecoyGeneratorType.ONE2MANY:
                for i in range(n):
                    filename_out = f"{filename}.{generator}.{i}{extension}"
                    write_fasta_file(filename_out, generator.convert_fasta(read_fasta_file(target_filename)))
