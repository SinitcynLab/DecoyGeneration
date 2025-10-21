from typing import Iterator, List

from src.decoy_generators.decoy_generator import DecoyGenerator


class DiannGenerator(DecoyGenerator):
    translation: dict[str, str] = {
        a: b
        for a, b in zip(
            list('GAVLIFMPWSCTYHKRQEND'),
            list('LLLVVLLLLTSSSSLLNDQE')
        )
    }

    def __str__(self):
        return "diann"

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
