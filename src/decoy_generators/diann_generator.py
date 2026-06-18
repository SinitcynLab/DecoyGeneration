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

    def _get_mutation_positions(self, peptide_length: int, flexible_range: range) -> List[int]:
        if peptide_length <= 1:
            return []

        if self.terminus == 'N':
            candidates = [0]
        else:
            candidates = [1, peptide_length - 2]

        positions: List[int] = []
        for pos in candidates:
            if pos < 0 or pos >= peptide_length:
                continue
            if self.terminus == 'C' and pos == peptide_length - 1:
                continue
            if pos not in flexible_range:
                continue
            if pos not in positions:
                positions.append(pos)
        return positions

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        for target in targets:
            new_sequence = []
            for peptide in self.protease.cleave(target):
                sequence = list(peptide.sequence)
                for pos in self._get_mutation_positions(len(sequence), peptide.flexible_range):
                    old = sequence[pos]
                    if old in self.translation:
                        sequence[pos] = self.translation[old]
                new_sequence.append("".join(sequence))
            yield "".join(new_sequence)
