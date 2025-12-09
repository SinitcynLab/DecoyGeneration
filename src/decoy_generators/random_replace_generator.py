from random import Random
from typing import Iterator, List

from src.decoy_generators.decoy_generator import DecoyGenerator


class RandomReplaceGenerator(DecoyGenerator):
    def __init__(self, special_amino_acids: List[str], random: Random):
        DecoyGenerator.__init__(self, special_amino_acids)
        self.random = random

    def __str__(self):
        return f"random_replace"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            positions: List[int] = list(self.get_positions_special_aas(target))
            sequence: List[str] = list(target)
            for idx in range(1, len(positions)):
                a: int = positions[idx - 1] + 1
                b: int = positions[idx]
                if b - a < 1: continue
                random_pos = self.random.randrange(a, b)
                sequence[random_pos] = self.random.choice(self.valid_aa_choices)
            yield "".join(sequence)
