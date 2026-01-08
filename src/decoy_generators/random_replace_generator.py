from random import Random
from typing import Iterator, List, Set

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType

class RandomReplaceGenerator(DecoyGenerator):
    decoy_generation_type: DecoyGeneratorType = DecoyGeneratorType.ONE2MANY
    
    def __init__(self, special_amino_acids: List[str], random: Random):
        DecoyGenerator.__init__(self, special_amino_acids)
        self.random = random
        aa_choices_set: Set[str] = set(self.canonical_amino_acids) - set(self.special_amino_acids)
        self.valid_aa_choices: List[str] = list(aa_choices_set)

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
                # prevent same-mass aa AND prevent putting back the same AA                
                sequence[random_pos] = self.random.choice(self.valid_aa_choices)
            yield "".join(sequence)
