import torch

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.base_smart_masking_esm import BaseSmartMaskingEsmGenerator

from torch import Tensor
    
class MaxProbMaskingEsmGenerator(BaseSmartMaskingEsmGenerator):
    def _get_score_and_token_choice(self, probs: Tensor, pos: int, original_aa: str):
        # find the top probability of aa's:
        token_prob, amino_acid_indices = torch.topk(probs[0, pos, self.aa_ids], k=self.k, largest=True)
        # convert list of tokens (indices of self.canonical_amino_acids) to a list of strings:
        amino_acids = self.canonical_amino_acids[amino_acid_indices]
        # the token probability is the score in this case
        score, aa_choice = self._get_feasible_token_with_max_score(original_aa, token_prob, amino_acids)
        # return feasible token with highest score and corresponding score:
        return score, aa_choice
    
    def __str__(self):
        return f"max_prob_{super().__str__()}"