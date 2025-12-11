import torch

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.esm_generator import EsmGenerator
from typing import List, Iterator, Tuple, NamedTuple

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
    ):
        EsmGenerator.__init__(self, local_path, random, special_amino_acids, sort_optimization,
                             batch_size, ml_generator_type, device, MaskingType.COUNT, 0, 1)
        self.k: int = len(self.special_amino_acids) + 2
        self.aa_ids = self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        
    def __str__(self):
        param_count = self.local_path.split('/')[-1].split('_')[2]
        return f"smart_masking_esm_{param_count}"
            
    def __get_all_peptides(self, sequence: str):
        positions: List[int] = list(self.get_positions_special_aas(sequence))
        for i in range(1, len(positions)):
            start: int = positions[i - 1] + 1
            end: int = positions[i]
            if start == end:
                continue
            else:
                yield range(start, end)
    
    def _get_current_val_aa_ids(self, og_aa_id: int):
        # get all 'valid' aa's that could fill the spot (exclude special aas and the original aa):
        current_valid_aa_ids = list(self.valid_aa_ids)
        if og_aa_id in current_valid_aa_ids: current_valid_aa_ids.remove(og_aa_id)
        return current_valid_aa_ids

    def _get_score_and_token_choice(self, probs: Tensor, position: int, original_aa: str) -> Tuple[float, Tensor]:
        raise NotImplementedError()
        
    def _batch_convert(self, target_batch: List[str]) -> Iterator[str]:
        for sequence in target_batch:
            max_pos_and_choices: List[Tuple[int, NamedTuple[Tensor, Tensor]]] = []
            # Tokenize sequence:
            input = self.tokenizer(sequence, return_tensors="pt", padding=True)
            input.to(self.device)
            # Save original input ids:
            modified_input_ids = torch.clone(input["input_ids"])
            for peptide in self.__get_all_peptides(sequence):
                max_score: float = -torch.inf
                max_score_pos: int = 0
                token_choice_at_max = None
                for pos in peptide:
                    input["input_ids"] = torch.clone(modified_input_ids)
                    # Mask out  position in the peptide:
                    input["input_ids"][0][pos] = self.tokenizer.mask_token_id
                    # obtain token probabilities:
                    with torch.no_grad():
                        outputs = self.model(**input)
                    probs: Tensor = torch.softmax(outputs.logits, dim=-1)
                    # compute difference between top-2 most likely tokens:
                    score, token_choice = self._get_score_and_token_choice(probs, pos, sequence[pos])
                    # if new smallest found, save position and choice:
                    if score > max_score:
                        max_score = score
                        max_score_pos = pos
                        token_choice_at_max = token_choice
                # save position and token choice for this peptide:
                max_pos_and_choices.append((max_score_pos, token_choice_at_max))

                # we now have the position and token choice for this peptide
                # we immediately put in the most-easily substituted aa and then proceed to next peptide, 
                # taking this new aa into account:
                modified_input_ids[0][max_score_pos] = token_choice_at_max

            new_sequence: List[str] = list(sequence)
            for mask_position, token_choice in max_pos_and_choices:
                new_sequence[mask_position] = self.canonical_amino_acids[token_choice]
            with open(f'token_choices_{self}.txt', 'a') as file:
                for _, token_choice in max_pos_and_choices:
                    file.write(f"{token_choice}\n")

            yield "".join(new_sequence)

    # should be in parent class, but kept it here because I'm not 1000% sure of correctness
    def _get_first_feasible_index(self, original_aa: str, top_tokens: Tensor):
        for idx in top_tokens:
            new_aa: str = self.canonical_amino_acids[idx]
            if new_aa == original_aa:
                continue
            if new_aa in self.special_amino_acids:
                continue
            if (new_aa == 'I' and original_aa == 'L') or (
                    new_aa == 'L' and original_aa == 'I'):
                continue
            return idx
        # if no aa satisfies constraints, default to original (esm gen itself does this too):
        return self.canonical_amino_acids.index(original_aa)
    
    def _get_feasible_token_with_max_score(self, original_aa: str, scores: Tensor, tokens: Tensor):
        sort_idx = torch.argsort(scores, descending=True)
        token_choice = self._get_first_feasible_index(original_aa, tokens[sort_idx])
        return scores[torch.where(tokens == token_choice)[0]], token_choice