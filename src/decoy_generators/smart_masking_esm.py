import torch

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.base_smart_masking_esm import BaseSmartMaskingEsmGenerator
from typing import List, Iterator, Tuple, NamedTuple

from random import Random
from torch import Tensor

class RelDiffMaskingEsmGenerator(BaseSmartMaskingEsmGenerator):
    def _get_score_and_token_choice(self, probs: Tensor, position: int, original_aa: str) -> Tuple[float, Tensor]:
        # get the original aa's id and its prob:
        og_aa_id, og_prob = self._get_og_aa_id_and_prob(probs, position, original_aa)
        # get the list of all valid aa id's in this context:
        current_valid_aa_ids = self._get_current_val_aa_ids(og_aa_id)
        
        # find the top probability of valid aa's:
        token_prob, token_choice = torch.topk(probs[0, position, current_valid_aa_ids], k=1, largest=True)
        score = - ((og_prob - token_prob[0]) / og_prob) # smaller relative difference means larger score
        return score, token_choice[0]
    
class MassMaskingEsmGenerator(BaseSmartMaskingEsmGenerator):
    def _get_score_and_token_choice(self, probs: Tensor, position: int, original_aa: str):
        # get the original aa's id and its prob:
        og_aa_id, _ = self._get_og_aa_id_and_prob(probs, position, original_aa)
        # get the list of all valid aa id's in this context:
        current_valid_aa_ids = self._get_current_val_aa_ids(og_aa_id)

        # find the top probability of valid aa's:
        token_prob, token_choice = torch.topk(probs[0, position, current_valid_aa_ids], k=1, largest=True)
        score = token_prob # the score is the mass of the substitutable aa with largest mass
        return score, token_choice[0]