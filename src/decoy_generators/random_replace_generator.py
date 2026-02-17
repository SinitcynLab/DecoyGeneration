from random import Random
from typing import Iterator, List, Set

from sympy import sequence

from src.decoy_generators.decoy_generator import DecoyGenerator, DecoyGeneratorType

from src.proteins.protease import Protease


class RandomReplaceGenerator(DecoyGenerator):
    decoy_generation_type: DecoyGeneratorType = DecoyGeneratorType.ONE2MANY
    
    def __init__(self, protease: Protease, random: Random):
        super().__init__(protease)
        self.random = random

    def __str__(self):
        return f"random_replace"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            new_sequence = []
            for peptide in self.protease.cleave(target):
                if len(peptide) > 1:
                    sequence = list(peptide.sequence)
                    random_pos = self.random.randint(0, len(peptide) - 1)
                    allowed_replacements = peptide.allowed_replacements[random_pos].copy()

                    # Do not replace amino acid with itself
                    allowed_replacements.discard(sequence[random_pos])
                    # Forbid I<->L replacements since they are indistinguishable in MS
                    if sequence[random_pos] == 'L':
                        allowed_replacements.discard('I')
                    elif sequence[random_pos] == 'I':
                        allowed_replacements.discard('L')

                    if len(allowed_replacements) > 0:
                        sequence[random_pos] = self.random.choice(list(allowed_replacements))
                    new_sequence.append("".join(sequence))

            yield "".join(new_sequence)
