import torch

from typing import List, Iterable
from torch import Tensor

from src.encoders.peptide_encoder import PeptideEncoder
from src.proteins.protease import Protease

class CustomTokenizer(PeptideEncoder):
    def __init__(self, amino_acids: List[str], protease: Protease):
        self.protease = protease
        self.itos = ["<pad>", "<unk>"] + amino_acids
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.vocab_size = len(self.itos)

    def __encode_peptide(self, peptide: str) -> Tensor:
        return torch.tensor([self.stoi.get(aa, self.unk_id) for aa in peptide])
    
    def __encode_protein(self, protein: str) -> List[Tensor]:
        out = []
        for peptide in self.protease.cleave(protein):
            out.append(self.__encode_peptide(peptide.sequence))
        return out
    
    def __call__(self, proteins: Iterable[str]) -> List[Tensor]:
        out = []
        for protein in proteins:
            out = out + self.__encode_protein(protein)
        return out