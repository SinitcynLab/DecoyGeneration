from typing import Iterator, List

from src.decoy_generators.decoy_generator import DecoyGenerator


class ReverseGenerator(DecoyGenerator):

    def __str__(self):
        return "reverse"

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            positions: List[int] = list(self.get_positions_special_aas(target))
            sequence: List[str] = list(target)
            for idx in range(1, len(positions)):
                a: int = positions[idx - 1] + 1
                b: int = positions[idx]
                if b - a < 2: continue
                for i in range(0, (b - a) // 2):
                    sequence[a + i], sequence[b - i - 1] = sequence[b - i - 1], sequence[a + i]
            yield "".join(sequence)
