import itertools
from enum import Enum
from typing import Iterator, List

from src.io.fasta import FastaRecord

from src.proteins.protease import Protease


class DecoyGeneratorType(Enum):
    ONE2ONE = 1,
    ONE2MANY = 2


class DecoyGenerator:
    canonical_amino_acids: List[str] = list("ACDEFGHIKLMNPQRSTVWY")
    decoy_generation_type: DecoyGeneratorType = DecoyGeneratorType.ONE2ONE


    def __init__(self, protease: Protease):
        self.protease = protease

    def convert(self, sequences: Iterator[str]) -> Iterator[str]:
        raise NotImplementedError()

    def convert_fasta(self, targets: Iterator[FastaRecord]) -> Iterator[FastaRecord]:
        targets1, targets2 = itertools.tee(targets)
        return (
            FastaRecord(head=record.head, sequence=new_seq)
            for record, new_seq in zip(targets2, self.convert(record.sequence for record in targets1))
        )
