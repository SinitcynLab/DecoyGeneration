from typing import Iterator, List

from src.decoy_generators.decoy_generator import DecoyGenerator

from src.proteins.protease import Protease


class HardcoreGenerator(DecoyGenerator):
    def __init__(self, protease: Protease):
        super().__init__(protease)

    def __str__(self):
        return "hardcore"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            new_sequence: List[str] = []
            for peptide in self.protease.cleave(target):
                flexible_range = peptide.flexible_range
                constant_prefix = peptide.sequence[:flexible_range.start]
                constant_suffix = peptide.sequence[flexible_range.stop:]
                mutable_part = peptide.sequence[flexible_range.start:flexible_range.stop]
                if "I" in mutable_part:
                    mutable_part = mutable_part.replace("I", "L", 1)
                elif "L" in mutable_part:
                    mutable_part = mutable_part.replace("L", "I", 1)
                elif "G" in mutable_part:
                    mutable_part = mutable_part.replace("G", "NN", 1)
                elif "NN" in mutable_part:
                    mutable_part = mutable_part.replace("NN", "G", 1)
                else:
                    idx = 0
                    while idx + 1 < len(mutable_part) and mutable_part[idx] == mutable_part[idx + 1]:
                        idx += 1
                    if idx + 1 < len(mutable_part):
                        mutable_part = mutable_part[:idx] + mutable_part[idx + 1] + mutable_part[idx] + mutable_part[idx + 2:]
                new_peptide = constant_prefix + "".join(mutable_part) + constant_suffix
                new_sequence.append(new_peptide)
            yield "".join(new_sequence)
