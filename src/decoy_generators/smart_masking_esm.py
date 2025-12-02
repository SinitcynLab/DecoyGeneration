import torch

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.esm_generator import EsmGenerator
from typing import List, Iterator, Tuple, NamedTuple

from time import time
from random import Random
from torch import Tensor

class SmartMaskingEsmGenerator(EsmGenerator):
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
        special_aa_ids = self.tokenizer.convert_tokens_to_ids(self.special_amino_acids)
        self.canonical_aa_ids = self.tokenizer(self.canonical_amino_acids)
        self.valid_aa_ids = [idx for idx in self.canonical_aa_ids if idx not in special_aa_ids]
        
    def __str__(self):
        param_count = self.local_path.split('/')[-1].split('_')[2]
        return f"smart_masking_esm_{param_count}.[{self.mask_percent}]"
            
    def __get_all_peptides(self, sequence: str):
        positions: List[int] = list(self.get_positions_special_aas(sequence))
        for i in range(1, len(positions)):
            start: int = positions[i - 1] + 1
            end: int = positions[i]
            if start == end:
                continue
            else:
                yield range(start, end)

    def _get_diff_and_token_choice(self, probs: Tensor, position: int, original_aa: str) -> Tuple[float, Tensor]:
        # get the original aa's id and its prob:
        og_aa_id = self.tokenizer.convert_tokens_to_ids(original_aa)
        og_prob = probs[0, position, og_aa_id]# get all 'valid' aa's that could fill the spot (exclude special aas and the original aa):
        # remove current original aa from valid aa's:
        current_valid_aa_ids = self.valid_aa_ids
        if og_aa_id in current_valid_aa_ids: current_valid_aa_ids.remove(og_aa_id)
        # find the top probability of valid aa's:
        token_prob, token_choice = torch.topk(probs[0, position, current_valid_aa_ids], k=1, largest=True)
        rel_diff = (og_prob - token_prob[0]) / og_prob
        return rel_diff, token_choice[0]
        
    def _batch_convert(self, target_batch: List[str]) -> Iterator[str]:
        for sequence in target_batch:
            min_pos_and_choices: List[Tuple[int, NamedTuple[Tensor, Tensor]]] = []
            # Tokenize sequence:
            input = self.tokenizer(sequence, return_tensors="pt", padding=True)
            input.to(self.device)
            # Save original input ids:
            modified_input_ids = torch.clone(input["input_ids"])
            for peptide in self.__get_all_peptides(sequence):
                min_rel_diff: float = torch.inf
                min_pos: int = 0
                token_choice_at_min = None
                for pos in peptide:
                    input["input_ids"] = torch.clone(modified_input_ids)
                    # Mask out  position in the peptide:
                    input["input_ids"][0][pos] = self.tokenizer.mask_token_id
                    # obtain token probabilities:
                    with torch.no_grad():
                        outputs = self.model(**input)
                    probs: Tensor = torch.softmax(outputs.logits, dim=-1)
                    # compute difference between top-2 most likely tokens:
                    rel_diff, token_choice = self._get_diff_and_token_choice(probs, pos, sequence[pos])
                    # if new smallest found, save position and choice:
                    if rel_diff < min_rel_diff:
                        min_rel_diff = rel_diff
                        min_pos = pos
                        token_choice_at_min = token_choice
                # save position and token choice for this peptide:
                min_pos_and_choices.append((min_pos, token_choice_at_min))

                # we now have the position and token choice for this peptide
                # we immediately put in the most-easily substituted aa and then proceed to next peptide, 
                # taking this new aa into account:
                modified_input_ids[0][min_pos] = token_choice_at_min

            new_sequence: List[str] = list(sequence)
            for mask_position, token_choice in min_pos_and_choices:
                new_sequence[mask_position] = self.canonical_amino_acids[token_choice]

            yield "".join(new_sequence)

    # should be in parent class, but kept it here because I'm not 1000% sure of correctness
    def _select_token(self, original_aa: str, token_choice: Tensor):
        for idx in token_choice:
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