
from random import Random
from typing import Iterator, List

from src.decoy_generators.ml_generator import MlGenerator, MlGeneratorType

class DiannGenerator(MlGenerator):
    def __init__(
            self,
            local_path: str,
            random: Random,
            special_amino_acids: List[str],
            mask_percent: float = 0.3,  # should be between 0.0 and 1.0
            sort_optimization: bool = True,
            batch_size: int = 64,
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST
    ):
        MlGenerator.__init__(self, local_path, random, special_amino_acids, mask_percent, sort_optimization,
                             batch_size, ml_generator_type)

    def __str__(self):
        return f"Peptune.[{self.mask_percent}]"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            positions: List[int] = list(self.get_positions_special_aas(target))
            sequence: List[str] = list(target)
            for idx in range(1, len(positions)):
                a: int = positions[idx - 1] + 1
                b: int = positions[idx]
                if b - a < 1: continue
                sequence[b - 1] = self.translation[sequence[b - 1]]
            yield "".join(sequence)
