import torch
import numpy as np

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.base_smart_masking_esm import BaseSmartMaskingEsmGenerator
from typing import List, Iterator, Tuple, NamedTuple

from random import Random
from torch import Tensor

class RelDiffMaskingEsmGenerator(BaseSmartMaskingEsmGenerator):
    def _get_score_and_token_choice(self, probs: Tensor, position: int, original_aa: str) -> Tuple[float, Tensor]:
        # get the original aa's prob:
        # find the id of the originally present aa, and find the probability corresponding to this aa as judged by ESM:
        og_aa_id = self.tokenizer.convert_tokens_to_ids(original_aa)
        og_prob = probs[0, position, og_aa_id]
        
        # find the top probability of aa's:
        tokens_prob, tokens = torch.topk(probs[0, position, self.aa_ids], k=self.k, largest=True)
        scores = tokens_prob.clone()
        for i, _ in enumerate(tokens_prob):
            scores[i] = - ((og_prob - tokens_prob[i])) / og_prob # multiply by -1 because find the max score
        score, token_choice = self._get_feasible_token_with_max_score(original_aa, scores, tokens)
        sav_arr = torch.tensor((og_prob, probs[0, position, token_choice]))
        with open(f'prob_distr_{self}.txt', 'a') as file:
            np.savetxt(file, sav_arr.cpu().numpy())
            file.write('\n')
        return score, token_choice
    
    def __str__(self):
        return f"rel_diff_{super().__str__()}"
    
class MassMaskingEsmGenerator(BaseSmartMaskingEsmGenerator):
    def _get_score_and_token_choice(self, probs: Tensor, position: int, original_aa: str):
        # find the top probability of aa's:
        token_prob, tokens = torch.topk(probs[0, position, self.aa_ids], k=self.k, largest=True)
        score, token_choice = self._get_feasible_token_with_max_score(original_aa, token_prob, tokens) # the mass (token_prob) is the score in this case
        og_aa_id = self.tokenizer.convert_tokens_to_ids(original_aa)
        sav_arr = torch.tensor((probs[0, position, og_aa_id], probs[0, position, token_choice]))
        with open(f'prob_distr_{self}.txt', 'a') as file:
            np.savetxt(file, sav_arr.cpu().numpy())
            file.write('\n')
        return score, token_choice
    
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
        # find the top probability of valid aa's:
        token_prob, tokens = torch.topk(probs[0, position, self.aa_ids], k=self.k, largest=True)
        scores = token_prob.clone()
        for i, _ in enumerate(scores):
            scores[i] = scores[i] / self.freq_dict[self.canonical_amino_acids[tokens[i]]] # the score is the frequency-normalized probability mass
        return self._get_feasible_token_with_max_score(original_aa, scores, tokens)
    
    def __str__(self):
        return f"freq_{super().__str__()}"