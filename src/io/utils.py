from typing import Iterator, List
from src.io.fasta import FastaRecord

def split_targets(target_sequences: List[str]):
    N = len(target_sequences)//2
    targets = target_sequences[0:N]
    pretend_decoys = target_sequences[N:N*2]

    return targets, pretend_decoys

def remove_long_sequences(records: Iterator[FastaRecord], cap_length: int = 10_000) -> Iterator[FastaRecord]:
    return [rec for rec in records if len(rec.sequence) <= cap_length]
