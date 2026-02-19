from typing import Iterator, List

from src.decoy_generators.decoy_generator import DecoyGenerator

from src.proteins.protease import Protease


class DiannGenerator(DecoyGenerator):
    translation: dict[str, str] = {
        a: b
        for a, b in zip(
            list('GAVLIFMPWSCTYHKRQENDU'),
            list('LLLVVLLLLTSSSSLLNDQES')
        )
    }

    def __init__(self, protease: Protease, terminus: str = 'C'):
        super().__init__(protease)
        if terminus not in ['C', 'N']:
            raise ValueError("Terminus argument must be 'C' or 'N'.")
        self.terminus = terminus

    def __str__(self):
        return f"diann_{self.terminus}"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            new_sequence = []
            for peptide in self.protease.cleave(target):
                sequence = list(peptide.sequence)
                if len(sequence) > 1:
                    if self.terminus == 'C':
                        pos = len(sequence) - 2
                    else:
                        pos = 0
                    sequence[pos] = self.translation[sequence[pos]]
                new_sequence.append("".join(sequence))
            yield "".join(new_sequence)
