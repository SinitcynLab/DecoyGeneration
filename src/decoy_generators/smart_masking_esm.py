import torch
import numpy as np

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.base_smart_masking_esm import BaseSmartMaskingEsmGenerator
from typing import List, Iterator, Tuple, NamedTuple

from random import Random
from torch import Tensor
    
class MaxProbMaskingEsmGenerator(BaseSmartMaskingEsmGenerator):
    def _get_score_and_token_choice(self, probs: Tensor, pos: int, original_aa: str):
        # find the top probability of aa's:
        token_prob, tokens = torch.topk(probs[0, pos, self.aa_ids], k=self.k, largest=True)
        # the token probability is the score in this case
        score, token_choice = self._get_feasible_token_with_max_score(original_aa, token_prob, tokens)
        # return feasible token with highest score and corresponding score:
        return score, token_choice
    
    def __str__(self):
        return f"max_prob_{super().__str__()}"