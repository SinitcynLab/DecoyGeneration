from typing import Iterator, List

from src.decoy_generators.decoy_generator import DecoyGenerator

from src.proteins.protease import Protease


class ReverseGenerator(DecoyGenerator):
    def __init__(self, protease: Protease):
        super().__init__(protease)

    def __str__(self):
        return "reverse"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            new_sequence: List[str] = []
            for peptide in self.protease.cleave(target):
                flexible_range = peptide.flexible_range
                constant_prefix = peptide.sequence[:flexible_range[0]]
                constant_suffix = peptide.sequence[flexible_range[1]:]
                mutable_part = list(peptide.sequence[flexible_range[0]:flexible_range[1]])
                mutable_part.reverse()
                new_peptide = constant_prefix + "".join(mutable_part) + constant_suffix
                new_sequence.append(new_peptide)
            yield "".join(new_sequence)
