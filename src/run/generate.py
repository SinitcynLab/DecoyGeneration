import os
import numpy as np
from typing import List

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType
from src.decoy_generators.ml_generator import MlGenerator
from src.io.fasta import write_fasta_file, read_fasta_file
from src.io.utils import remove_long_sequences

def generate_decoys(target_file: str, generator: DecoyGenerator, n: int, destination_dir: str):
    filename, extension = os.path.splitext(target_file)
    filename = os.path.join(destination_dir, os.path.basename(filename))
    if issubclass(type(generator), MlGenerator):
        for i in range(n):
            filename_out = f"{filename}.{generator}.{i}{extension}"
            target_records = [record for record in read_fasta_file(target_file)]
            target_records = remove_long_sequences(target_records, cap_length=10_000)
            batch_starts = np.arange(0, len(target_records), generator.batch_size)
            for start in batch_starts:
                end = min(start + generator.batch_size, len(target_records))
                write_fasta_file(filename_out, generator.convert_fasta(target_records[start:end]), 60, 'a')
                print(f"{end}/{len(target_records)}")
    elif generator.decoy_generation_type == DecoyGeneratorType.ONE2ONE:
        filename_out = f"{filename}.{generator}{extension}"
        write_fasta_file(filename_out, generator.convert_fasta(read_fasta_file(target_file)))
    elif generator.decoy_generation_type == DecoyGeneratorType.ONE2MANY:
        for i in range(n):
            filename_out = f"{filename}.{generator}.{i}{extension}"
            write_fasta_file(filename_out, generator.convert_fasta(read_fasta_file(target_file)))