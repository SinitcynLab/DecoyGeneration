from random import Random
from typing import List, Iterator

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType


class ShuffleGenerator(DecoyGenerator):
    random: Random
    decoy_generation_type: DecoyGeneratorType = DecoyGeneratorType.ONE2MANY

    def __init__(self, special_amino_acids: List[str], random: Random, skip_prob : float = 0):
        self.random = random
        self.skip_prob = skip_prob
        super().__init__(special_amino_acids)

    def __str__(self):
        return f"shuffle._s{self.skip_prob}_"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            positions: List[int] = list(self.get_positions_special_aas(target))
            sequence: List[str] = list(target)

            for idx in range(1, len(positions)):
                a: int = positions[idx - 1] + 1
                b: int = positions[idx]
                if b - a < 2 or self.sample_pos_prob() < self.skip_prob: continue
                if b - a == 2:
                    sequence[a], sequence[a + 1] = sequence[a + 1], sequence[a]
                else:
                    ab: List[int] = list(range(a, b))
                    self.random.shuffle(ab)
                    for i, j in zip(range(a, b), ab):
                        sequence[i] = target[j]
            yield "".join(sequence)

    def sample_pos_prob(self) -> float:
        s = self.random.uniform(0,1)
        if s == 0:
            return self.sample_pos_prob()
        else:
            return s