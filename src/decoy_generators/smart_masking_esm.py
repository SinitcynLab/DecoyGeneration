import torch

from src.decoy_generators.base_smart_masking_esm import BaseSmartMaskingEsmGenerator

from torch import Tensor

from typing import Set
    
class MaxProbMaskingEsmGenerator(BaseSmartMaskingEsmGenerator):
    def _get_score_and_token_choice(
        self,
        probs: Tensor,
        pos: int,
        original_aa: str,
        allowed_replacements: Set[str]
    ):
        # find the top probability of aa's:
        token_prob, amino_acid_indices = torch.topk(probs[0, pos, self.aa_ids], k=self.k, largest=True)
        # convert list of tokens (indices of self.canonical_amino_acids) to a list of strings:
        amino_acids = [self.canonical_amino_acids[index] for index in amino_acid_indices]
        # the token probability is the score in this case
        score, aa_choice = self._get_feasible_token_with_max_score(original_aa, token_prob, amino_acids, allowed_replacements)
        # return feasible token with highest score and corresponding score:
        return score, aa_choice
    
    def __str__(self):
        return f"max_prob_{super().__str__()}"
    
class RelDiffMaskingEsmGenerator(BaseSmartMaskingEsmGenerator):
    def _get_score_and_token_choice(
        self,
        probs: Tensor,
        position: int,
        original_aa: str,
        allowed_replacements: Set[str]
    ):
        # get the original aa's prob:
        # find the id of the originally present aa, and find the probability corresponding to this aa as judged by ESM:
        og_aa_id = self.tokenizer.convert_tokens_to_ids(original_aa) # note that this id is not the same as the index of the aa in 'self.canonical_amino_acids'
        og_prob = probs[0, position, og_aa_id]
        
        tokens_prob, amino_acid_indices = torch.topk(probs[0, position, self.aa_ids], k=self.k, largest=True)
        amino_acids = [self.canonical_amino_acids[index] for index in amino_acid_indices]
        
        scores = tokens_prob.clone()
        for i, _ in enumerate(tokens_prob):
            scores[i] = - ((og_prob - tokens_prob[i])) / og_prob # multiply by -1 because find the max score
        
        score, aa_choice = self._get_feasible_token_with_max_score(original_aa, scores, amino_acids, allowed_replacements)
        return score, aa_choice
    
    def __str__(self):
        return f"rel_diff_{super().__str__()}"