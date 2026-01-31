import torch

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.esm_generator import EsmGenerator
from typing import List, Iterator, Tuple

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

    def _get_score_and_token_choice(self, probs: Tensor, pos: int, original_aa: str) -> Tuple[float, Tensor]:
        raise NotImplementedError()
        
    def _batch_convert(self, target_batch: List[str]) -> Iterator[str]:
        for sequence in target_batch:
            max_pos_and_choices: List[Tuple[int, str]] = []
            # Tokenize sequence:
            input = self.tokenizer(sequence, return_tensors="pt", padding=True)
            input.to(self.device)
            # Save original input ids:
            original_input_ids: Tensor = torch.clone(input["input_ids"])
            for peptide in self.get_all_peptides(sequence):
                max_score: float = -torch.inf
                max_score_pos: int = 0
                aa_choice_at_max: str = ""
                for pos in peptide:
                    input["input_ids"] = torch.clone(original_input_ids)
                    # Mask out  position in the peptide:
                    input["input_ids"][0][pos + 1] = self.tokenizer.mask_token_id # increment by one to account for [cls]
                    # obtain token probabilities:
                    with torch.no_grad():
                        outputs = self.model(**input)
                    probs: Tensor = torch.softmax(outputs.logits, dim=-1)
                    probs = probs[:, 1:, :] # drop the first dimension (batch_size is always 1 here) and drop entries corresponding to [cls]
                    # compute score:
                    score, aa_choice = self._get_score_and_token_choice(probs, pos, sequence[pos])
                    # if new best found, save position and choice:
                    if score > max_score:
                        max_score = score
                        max_score_pos = pos
                        aa_choice_at_max = aa_choice
                # save position and token choice for this peptide:
                max_pos_and_choices.append((max_score_pos, aa_choice_at_max))

            new_sequence: List[str] = list(sequence)
            for mask_position, aa_choice in max_pos_and_choices:
                new_sequence[mask_position] = aa_choice

            yield "".join(new_sequence)
    
    def _get_feasible_token_with_max_score(self, original_aa: str, scores: Tensor, amino_acids: List[str]):
        sort_idx = torch.argsort(scores, descending=True) # practically, amino acids are already sorted. This is for clarity/robustness

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
        return scores[pos_aa_choice], aa_choice
    
    def _update_input_ids(self, input_ids: Tensor, max_score_pos: int, aa_choice_at_max: str):
        # get the id of the token corresponding to the best amino acid (different than its index in self.canonical_amino_acids):
        token_choice_at_max_id: int = self.tokenizer.convert_tokens_to_ids(aa_choice_at_max)
        # update the token in the amino acid chain:
        input_ids[0][max_score_pos] = token_choice_at_max_id
        # return the updated tokenization:
        return input_ids