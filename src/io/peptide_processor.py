from typing import List, Iterator

class PeptideProcessor(object):
    def __init__(self, special_amino_acids: List[str]):
        object.__init__(self)
        self.special_amino_acids = special_amino_acids

    def get_all_peptides(self, sequence: str):
        # Returns integer ranges, together coinciding with all indices of string characters WITHIN the sequence's peptides
        # (I.e. excluding the special amino acids)
        positions: List[int] = list(self.get_positions_special_aas(sequence))
        for i in range(1, len(positions)):
            start: int = positions[i - 1] + 1
            end: int = positions[i]
            if start == end:
                continue
            else:
                yield range(start, end)

    def get_positions_special_aas(self, sequence: str) -> Iterator[int]:
        if sequence[0] == "M":  # special case of the first amino acid in proteins, which is usually M
            yield 0
        else:
            yield -1
        for idx, aa in enumerate(sequence):
            if aa in self.special_amino_acids:
                yield idx
        yield len(sequence)