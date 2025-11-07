import math
from enum import Enum

import torch
from torch import Tensor

from src.decoy_generators.decoy_generator import DecoyGeneratorType
from src.decoy_generators.ml_generator import MlGenerator, MlGeneratorType
from typing import Iterator, List, Tuple
from transformers import EsmTokenizer, EsmForMaskedLM

from random import Random

torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EsmGenerator(MlGenerator):
    model: EsmForMaskedLM
    tokenize: EsmTokenizer
    random: Random
    mask_percent: float
    sort_optimization: bool
    batch_size: int
    esm_generator_type: MlGeneratorType

    decoy_generation_type: DecoyGeneratorType = DecoyGeneratorType.ONE2MANY

    def __init__(
            self,
            local_path: str,
            random: Random,
            special_amino_acids: List[str],
            mask_percent: float = 0.3,  # should be between 0.0 and 1.0
            sort_optimization: bool = True,
            batch_size: int = 64,
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST
    ):
        MlGenerator.__init__(self, local_path, random, special_amino_acids, mask_percent, sort_optimization,
                             batch_size, ml_generator_type)
        self.model = EsmForMaskedLM.from_pretrained(local_path, local_files_only=True)
        self.tokenizer = EsmTokenizer.from_pretrained(local_path, local_files_only=True)
        self.model.eval()

    def __str__(self):
        param_count = self.local_path.split('/')[-1].split('_')[2]
        return f"esm{param_count}.{self.esm_generator_type.name.lower()}.[{self.mask_percent}]"

    def _batch_convert(self, target_batch: List[str]) -> Iterator[str]:
        aa_ids = self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)

        k: int = 2 + len(self.special_amino_acids)  # why 2 - aa itself and I/L dillema

        inputs = self.tokenizer(target_batch, return_tensors="pt", padding=True)  # [batch_size, L, vocab]
        mask_positions: List[List[int]] = [[] for _ in range(len(target_batch))]
        for sequence_idx, sequence in enumerate(target_batch):
            for mask_idx in self._get_masked_positions(sequence):
                inputs["input_ids"][sequence_idx][mask_idx] = self.tokenizer.mask_token_id
                mask_positions[sequence_idx].append(mask_idx)

        with torch.no_grad():
            outputs = self.model(**inputs)
        probs: Tensor = torch.softmax(outputs.logits, dim=-1)  # [batch_size, L, vocab]

        for sequence_idx, sequence in enumerate(target_batch):
            new_sequence: List[str] = list(sequence)
            for mask_position in mask_positions[sequence_idx]:
                top_idx: Tensor = None
                match self.esm_generator_type:
                    case MlGeneratorType.BEST:
                        _, top_idx = torch.topk(probs[sequence_idx, mask_position, aa_ids], k=k, largest=True)
                    case MlGeneratorType.WORST:
                        _, top_idx = torch.topk(probs[sequence_idx, mask_position, aa_ids], k=k, largest=False)

                original_aa: str = sequence[mask_position]
                for idx in top_idx:
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
