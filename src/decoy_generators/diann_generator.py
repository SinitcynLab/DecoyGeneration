from typing import Iterator, List

from src.decoy_generators.decoy_generator import DecoyGenerator


class DiannGenerator(DecoyGenerator):
    translation: dict[str, str] = {
        a: b
        for a, b in zip(
            list('GAVLIFMPWSCTYHKRQENDU'),
            list('LLLVVLLLLTSSSSLLNDQES')
        )
    }

    def __init__(self, special_amino_acids: List[str], terminus: str = 'C'):
        DecoyGenerator.__init__(self, special_amino_acids)
        if terminus not in ['C', 'N']:
            raise ValueError("Terminus argument must be 'C' or 'N'.")
        self.terminus = terminus

    def __str__(self):
        return f"diann_{self.terminus}"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            positions: List[int] = list(self.get_positions_special_aas(target))
            sequence: List[str] = list(target)
            for idx in range(1, len(positions)):
                a: int = positions[idx - 1] + 1
                b: int = positions[idx]
                if b - a < 1: continue
                if self.terminus == 'C':
                    pos = b - 1
                else:
                    pos = a
                sequence[pos] = self.translation[sequence[idx]]
            yield "".join(sequence)
