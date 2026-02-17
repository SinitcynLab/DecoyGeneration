from random import Random
from typing import List, Iterator

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType

from src.proteins.protease import Protease


class ShuffleGenerator(DecoyGenerator):
    random: Random
    decoy_generation_type: DecoyGeneratorType = DecoyGeneratorType.ONE2MANY

    def __init__(self, protease: Protease, random: Random, skip_prob: float = 0):
        self.random = random
        self.skip_prob = skip_prob
        super().__init__(protease)

    def __str__(self):
        skip_prob: str = f"{self.skip_prob}".replace(".", "")
        return f"shuffle.s{skip_prob}"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            new_sequence: List[str] = []
            for peptide in self.protease.cleave(target):
                flexible_range = peptide.flexible_range
                if len(flexible_range) == 0:
                    new_sequence.append(peptide.sequence)
                    continue
                constant_prefix = peptide.sequence[:flexible_range.start]
                constant_suffix = peptide.sequence[flexible_range.stop:]
                mutable_part = list(peptide.sequence[flexible_range.start:flexible_range.stop])
                if len(mutable_part) > 1 and self.random.uniform(0, 1) >= self.skip_prob:
                    if len(mutable_part) == 2:
                        mutable_part[0], mutable_part[1] = mutable_part[1], mutable_part[0]
                    else:
                        self.random.shuffle(mutable_part)
                new_peptide = constant_prefix + "".join(mutable_part) + constant_suffix
                new_sequence.append(new_peptide)
            yield "".join(new_sequence)
