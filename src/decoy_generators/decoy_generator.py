import itertools
from enum import Enum
from typing import Iterator, List

from src.io.fasta import FastaRecord


class DecoyGeneratorType(Enum):
    ONE2ONE = 1,
    ONE2MANY = 2


class DecoyGenerator(object):
    canonical_amino_acids: List[str] = list("ACDEFGHIKLMNPQRSTVWY")
    decoy_generation_type: DecoyGeneratorType = DecoyGeneratorType.ONE2ONE

    special_amino_acids: List[str]

    def __init__(self, special_amino_acids: List[str]):
        self.special_amino_acids = special_amino_acids

    def convert(self, sequences: Iterator[str]) -> Iterator[str]:
        raise NotImplementedError()

    def convert_fasta(self, targets: Iterator[FastaRecord]) -> Iterator[FastaRecord]:
        targets1, targets2 = itertools.tee(targets)
        return (
            FastaRecord(head=record.head, sequence=new_seq)
            for record, new_seq in zip(targets2, self.convert(record.sequence for record in targets1))
        )

    def get_positions_special_aas(self, sequence: str) -> Iterator[int]:
        if sequence[0] == "M":  # special case of the first amino acid in proteins, which is usually M
            yield 0
        else:
            yield -1
        for idx, aa in enumerate(sequence):
            if aa in self.special_amino_acids:
                yield idx
        yield len(sequence)
