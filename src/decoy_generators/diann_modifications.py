from random import Random

from typing import List, Set
from src.decoy_generators.diann_generator import DiannGenerator

class DiannRandomPos(DiannGenerator):
    def __init__(self, special_amino_acids: List[str], random: Random):
        DiannGenerator.__init__(self, special_amino_acids)
        self.random = random
    
    def __str__(self):
        return "diann_random_pos"
    
    def convert(self, targets):
        for target in targets:
            positions: List[int] = list(self.get_positions_special_aas(target))
            sequence: List[str] = list(target)
            for idx in range(1, len(positions)):
                a: int = positions[idx - 1] + 1
                b: int = positions[idx]
                if b - a < 1: continue
                random_pos = self.random.randrange(a, b)
                sequence[random_pos] = self.translation[sequence[random_pos]]
            yield "".join(sequence)

class DiannRandomAcid(DiannGenerator):
    def __init__(self, special_amino_acids: List[str], random: Random):
        DiannGenerator.__init__(self, special_amino_acids)
        self.random = random
        aa_choices_set: Set[str] = set(self.canonical_amino_acids) - set(self.special_amino_acids)
        self.valid_aa_choices: List[str] = list(aa_choices_set)
    
    def __str__(self):
        return "diann_random_acid"
    
    def convert(self, targets):
        for target in targets:
            positions: List[int] = list(self.get_positions_special_aas(target))
            sequence: List[str] = list(target)
            for idx in range(1, len(positions)):
                a: int = positions[idx - 1] + 1
                b: int = positions[idx]
                if b - a < 1: continue
                sequence[b-a] = self.random.choice(self.valid_aa_choices)
            yield "".join(sequence)
    