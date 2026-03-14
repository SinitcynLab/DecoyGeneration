import torch

from typing import List, Iterable
from torch import Tensor

from src.encoders.peptide_encoder import PeptideEncoder
from src.proteins.protease import Protease

class CustomTokenizer(PeptideEncoder):
    def __init__(self, amino_acids: List[str], protease: Protease, peptide_level: bool = False):
        self.protease = protease
        self.itos = ["<pad>", "<unk>"] + amino_acids
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.vocab_size = len(self.itos)
        self.peptide_level = peptide_level
        self.peptide_memory = set()
        self.special_amino_acids = ["K", "R"]

    def __encode_sequence(self, peptide: str) -> Tensor:
        return torch.tensor([self.stoi.get(aa, self.unk_id) for aa in peptide])
    
    def get_positions_special_aas(self, protein: str):
        if protein[0] == "M":
            yield 0
        else:
            yield -1
        for idx, aa in enumerate(protein):
            if aa in self.special_amino_acids:
                yield idx
        if protein[-1] not in self.special_amino_acids:
            yield len(protein) - 1

    def get_all_peptides(self, protein: str):
        positions: List[int] = list(self.get_positions_special_aas(protein))
        for i in range(1, len(positions)):
            start: int = positions[i - 1] + 1
            end: int = positions[i]
            if start == end:
                continue
            else:
                yield (start, end)
    
    def __encode_protein_peptide_level(self, protein: str) -> List[Tensor]:
        out = []
        for a, b in self.get_all_peptides(protein):
            sequence = protein[a:(b+1)]
            # only encode UNIQUE peptides:
            if sequence in self.peptide_memory:
                continue
            else:
                self.peptide_memory.add(sequence)
            out.append(self.__encode_sequence(sequence))
        return out

    def reset(self):
        self.peptide_memory = set()

    def print_seq_stats(self, preface: str):
        print(preface)
        print("n.o. sequences: {0}".format(len(self.peptide_memory)))
        print("max_length: {0}".format(max(len(s) for s in self.peptide_memory)))
        print("min_length: {0}".format(min(len(s) for s in self.peptide_memory)))

    def get_num_peptides(self):
        return len(self.peptide_memory)
    
    def __encode_protein_sequence_level(self, protein: str) -> List[Tensor]:
        encoding = self.__encode_sequence(protein)
        return [encoding]
        
    def __call__(self, proteins: Iterable[str]) -> List[Tensor]:
        out = []
        if self.peptide_level:
            for protein in proteins:
                out = out + self.__encode_protein_peptide_level(protein)
        else:
            for protein in proteins:
                out = out + self.__encode_protein_sequence_level(protein)
        return out