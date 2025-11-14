import torch

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.esm_generator import EsmGenerator
from typing import List, Iterator, Tuple, NamedTuple

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
        
    def _batch_convert(self, target_batch: List[str]) -> Iterator[str]:
        aa_ids = self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)

        k: int = 2 + len(self.special_amino_acids)  # why 2 - aa itself and I/L dillema

        for sequence in target_batch:
            pos_and_token_choice: List[Tuple[int, NamedTuple[Tensor, Tensor]]] = []
            # Tokenize sequence:
            input = self.tokenizer(sequence, return_tensors="pt", padding=True)
            input.to(self.device)
            # Save original input ids:
            og_input_ids = torch.clone(input["input_ids"])
            for peptide in self.__get_all_peptides(sequence):
                min_diff: float = torch.inf
                min_pos: int = 0
                token_choice_at_min = None
                for pos in peptide:
                    input["input_ids"] = torch.clone(og_input_ids)
                    # Mask out  position in the peptide:
                    input["input_ids"][0][pos] = self.tokenizer.mask_token_id
                    # obtain token probabilities:
                    with torch.no_grad():
                        outputs = self.model(**input)
                    probs: Tensor = torch.softmax(outputs.logits, dim=-1)
                    # compute difference between top-2 most likely tokens:
                    token_probs, token_choice = torch.topk(probs[0, pos, aa_ids], k=k, largest=True)
                    diff = token_probs[0] - token_probs[1]
                    # if new smallest found, save position and k most likely tokens:
                    if diff < min_diff:
                        min_diff = diff
                        min_pos = pos
                        token_choice_at_min = token_choice
                # save position and token choice for this peptide:
                pos_and_token_choice.append((min_pos, token_choice_at_min))

            new_sequence: List[str] = list(sequence)
            for mask_position, token_choice in pos_and_token_choice:
                original_aa: str = sequence[mask_position]
                for idx in token_choice:
                    new_aa: str = self.canonical_amino_acids[idx]
                    if new_aa == original_aa:
                        continue
                    if new_aa in self.special_amino_acids:
                        continue
                    if (new_aa == 'I' and original_aa == 'L') or (
                            new_aa == 'L' and original_aa == 'I'):
                        continue
                    new_sequence[mask_position] = new_aa
                    break
            yield "".join(new_sequence)