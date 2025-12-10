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
    
    def __str__(self):
        return f"rel_diff_{super().__str__()}"
    
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
    
    def __str__(self):
        return f"mass_{super().__str__()}"
    
class FreqMaskingEsmGenerator(BaseSmartMaskingEsmGenerator):
    def __init__(
            self,
            local_path: str,
            random: Random,
            special_amino_acids: List[str],
            sort_optimization: bool = True,
            batch_size: int = 64,
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST,
            device: torch.device = 'cpu',
    ):
        BaseSmartMaskingEsmGenerator.__init__(self, local_path, random, special_amino_acids, sort_optimization,
                             batch_size, ml_generator_type, device)
        self.freq_dict = {'M': 0.020864142557172387, 'S': 0.08988429349173019, 'Y': 0.03385337941440006, 
                          'T': 0.05914553936908157, 'D': 0.05835266995116823, 'N': 0.06157044656693329, 
                          'P': 0.043781098592239755, 'Q': 0.03949708452825461, 'K': 0.07337497514836089, 
                          'R': 0.04444664591384583, 'A': 0.05488875997810332, 'L': 0.09512220197778196, 
                          'V': 0.05556043510113596, 'H': 0.02172305605712745, 'E': 0.06520048804534029, 
                          'G': 0.0496671922958557, 'F': 0.04436460146141255, 'I': 0.06560458250608014, 
                          'W': 0.010402964221810071, 'C': 0.012695442822165755} # calculated based on UP000002311_559292.fasta

    def _get_score_and_token_choice(self, probs: Tensor, position: int, original_aa: str):
        # get the original aa's id and its prob:
        og_aa_id, _ = self._get_og_aa_id_and_prob(probs, position, original_aa)
        # get the list of all valid aa id's in this context:
        current_valid_aa_ids = self._get_current_val_aa_ids(og_aa_id)

        # find the top probability of valid aa's:
        token_prob, token_choice = torch.topk(probs[0, position, current_valid_aa_ids], k=len(current_valid_aa_ids), largest=True)
        scores = token_prob.clone()
        for i in enumerate(scores):
            scores[i] = scores[i] / self.freq_dict[self.canonical_amino_acids[token_choice]] # the score is the frequency-normalized probability mass
        choice_idx = torch.argmax(scores)
        return scores[choice_idx], token_choice[choice_idx]
    
    def __str__(self):
        return f"mass_{super().__str__()}"