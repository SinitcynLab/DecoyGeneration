import torch
import heapq

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.esm_generator import EsmGenerator
from typing import List, Tuple
from collections import deque

from time import time
from random import Random
from torch import Tensor

class BaseSmartMaskingEsmGenerator(EsmGenerator):
    def __init__(
            self,
            local_path: str,
            random: Random,
            special_amino_acids: List[str],
            sort_optimization: bool = True,
            batch_size: int = 64,
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST,
            device: torch.device = 'cpu',
            weight_type: torch.dtype = torch.float32
    ):
        EsmGenerator.__init__(self, local_path, random, special_amino_acids, sort_optimization,
                             batch_size, ml_generator_type, device, MaskingType.COUNT, 0, 1, weight_type)
        self.k: int = len(self.canonical_amino_acids) # you want to compute scores over all amino acids
        self.aa_ids: List[int] = self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        
    def __str__(self):
        param_count: str = self.local_path.split('/')[-1].split('_')[2]
        out: str = f"smart_masking_esm_{param_count}"
        
        if self.weight_type == torch.float16:
            out = out + ".16b"

        return out

    def _get_score(self, probs: Tensor, pos: int, original_aa: str) -> Tuple[float, Tensor]:
        raise NotImplementedError()
    
    def _get_masked_positions(self, sequence: str):
        positions: List[int] = []

        # get the token-level probabilities from the model, without masking anything:
        input = self.tokenizer(sequence, return_tensors="pt", padding=True)
        input.to(self.device)
        with torch.no_grad():
            outputs = self.model(**input)
        probs: Tensor = torch.softmax(outputs.logits, dim=-1)
        probs = probs[:, 1:, :] # drop entries corresponding to [cls] token

        # select optimal position(s) for each peptide:
        for peptide in self.get_all_peptides(sequence):
            pos_for_peptide: deque = deque(maxlen=self.mask_count)
            pos_for_peptide_scores: List[float] = [-torch.inf] * self.mask_count # initialize scores as 'mask_count' times the smallest possible value
            for pos in peptide:
                score = self._get_score(probs, pos, sequence[pos])
                if score > min(pos_for_peptide_scores) or len(pos_for_peptide) < self.mask_count:
                    heapq.heappushpop(pos_for_peptide_scores, score)
                    pos_for_peptide.append(pos)
            positions.extend(pos_for_peptide)

        return sorted(positions)
    
    def _get_highest_feasible_score(self, original_aa: str, scores: Tensor, amino_acids: List[str]):
        sort_idx = torch.argsort(scores, descending=True)

        aa_choice = 'A' # if no aa satisfies constraints, default to A (first amino acid in list)
        for new_aa in [amino_acids[idx] for idx in sort_idx]:
            if new_aa == original_aa:
                continue
            if new_aa in self.special_amino_acids:
                continue
            if (new_aa == 'I' and original_aa == 'L') or (
                    new_aa == 'L' and original_aa == 'I'):
                continue
            aa_choice = new_aa
            break
        
        # recover the index of aa_choice among the unsorted amino acid list:
        pos_aa_choice: int = amino_acids.index(aa_choice)
        # use aforementioned position to get score corresponding to chosen token:
        return scores[pos_aa_choice]