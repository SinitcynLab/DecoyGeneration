import torch

from scipy.stats import entropy
from src.experimental.decoy_generators.unmasked_smart_masker_base import UnmaskedSmartMaskingEsmGenerator

from torch import Tensor

class UnmaskedMaxProbSmartMasker(UnmaskedSmartMaskingEsmGenerator):
    def _get_score(self, probs: Tensor, pos: int, original_aa: str):
        # find the top probability of aa's:
        token_prob, amino_acid_indices = torch.topk(probs[0, pos, self.aa_ids], k=self.k, largest=True)
        # convert list of tokens (indices of self.canonical_amino_acids) to a list of strings:
        amino_acids = [self.canonical_amino_acids[index] for index in amino_acid_indices]
        # the token probability is the score in this case:
        score = self._get_highest_feasible_score(original_aa, token_prob, amino_acids)
        # return highest feasible score:
        return score
    
    def __str__(self):
        return f"max_prob_{super().__str__()}"
    
class UnmaskedMaxEntropySmartMasker(UnmaskedSmartMaskingEsmGenerator):
    def _get_score(self, probs, pos, original_aa):
        # get the distribution over feasible tokens:
        token_prob, _ = torch.topk(probs[0, pos, self.aa_ids], k=self.k, largest=True)
        # compute entropy of distribution (this scipy function normalizes by default):
        score = entropy(token_prob.cpu().numpy())
        # return entropy as score
        return score
    
    def __str__(self):
        return f"max_entropy_{super().__str__()}"