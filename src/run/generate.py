import os
import numpy as np
from typing import List

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType
from src.decoy_generators.ml_generator import MlGenerator
from src.io.fasta import write_fasta, read_fasta_file
from src.io.utils import remove_long_sequences
from tqdm import tqdm

def generate_decoys(target_file: str, generator: DecoyGenerator, n: int, destination_dir: str):
    filename, extension = os.path.splitext(target_file)
    filename = os.path.join(destination_dir, os.path.basename(filename))
    print(f"Generating output of {generator} for file {target_file}.")
    filename_out = f"{filename}.{generator}{extension}"
    with open(filename_out, "w") as filestream:
        if issubclass(type(generator), MlGenerator):
            for _ in range(n):
                target_records = list(read_fasta_file(target_file))
                target_records = remove_long_sequences(target_records, cap_length=10_000)
                batch_starts = np.arange(0, len(target_records), generator.batch_size)
                for start in tqdm(batch_starts):
                    end = min(start + generator.batch_size, len(target_records))
                    records = target_records[start:end]
                    write_fasta(filestream, records)
                    write_fasta(filestream, generator.convert_fasta(records), prefix="rev_")
        elif generator.decoy_generation_type == DecoyGeneratorType.ONE2ONE:
            records = list(read_fasta_file(target_file))
            write_fasta(filestream, records)
            write_fasta(filestream, generator.convert_fasta(records), prefix="rev_")
        elif generator.decoy_generation_type == DecoyGeneratorType.ONE2MANY:
            records = list(read_fasta_file(target_file))
            write_fasta(filestream, records)
            for _ in range(n):
                write_fasta(filestream, generator.convert_fasta(records), prefix="rev_")
    print(f"Decoys written to {filename_out}.")
