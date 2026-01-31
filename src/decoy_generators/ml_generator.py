import math
import torch
import numpy as np

from typing import Iterator, List, Tuple
from random import Random
from enum import Enum
from torch import Tensor

from src.decoy_generators.decoy_generator import DecoyGenerator

class MaskingType(Enum):
    COUNT = 1,
    PERCENT = 2

class MlGeneratorType(Enum):
    BEST = 1,
    WORST = 2

class MlGenerator(DecoyGenerator):
    def __init__(
            self,
            local_path: str,
            random: Random,
            special_amino_acids: List[str],
            sort_optimization: bool = True,
            batch_size: int = 64,
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST,
            device : torch.device = 'cpu',
            masking_type: MaskingType = MaskingType.PERCENT,
            mask_percent: float = 0.3,
            mask_count: int = 1,
            weight_type: torch.dtype = torch.float32
    ):
        DecoyGenerator.__init__(self, special_amino_acids)
        self.random = random
        self.sort_optimization = sort_optimization
        self.batch_size = batch_size
        self.ml_generator_type = ml_generator_type
        self.local_path = local_path
        self.device = device
        self.masking_type = masking_type
        self.mask_percent = mask_percent
        self.mask_count = mask_count
        self.weight_type = weight_type

    def _get_masked_positions(self, sequence: str):
        positions: List[int] = list(self.get_positions_special_aas(sequence))
        for idx in range(1, len(positions)):
            start: int = positions[idx - 1] + 1
            end: int = positions[idx]
            n: int = end - start
            m: int = self._get_masking_count(n)
            if m == 0:
                continue
            for i in sorted(self._select_mask(start, end, m)):
                yield i

    def _select_mask(self, start: int, end: int, size: int):
        return self.random.sample(range(start, end), size)

    def _get_masking_count(self, seq_len: int) -> int:
        if self.masking_type == MaskingType.PERCENT:
            return math.ceil(seq_len * self.mask_percent)
        elif self.masking_type == MaskingType.COUNT:
            return min(seq_len, self.mask_count)
        else:
            raise ValueError("No valid masking type has been set for generator.")

    @staticmethod
    def _batch(a: Iterator[str], batch_size: int) -> Iterator[List[str]]:
        b: List[str] = []
        n: int = 0
        for item in iter(a):
            b.append(item)
            n += 1
            if n % batch_size == 0:
                yield b
                b = []
        yield b

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        if self.sort_optimization:
            target_list: List[str] = list(targets)
            target_out_list: List[Tuple[int, str]] = []
            sort_idx: List[int] = sorted(range(0, len(target_list)), key=lambda idx: len(target_list[idx]))
            for i in range(0, len(target_list), self.batch_size):
                tmp: List[str] = [
                    _ for _ in self._batch_convert([target_list[idx] for idx in sort_idx[i: i + self.batch_size]])
                ]
                for idx, record in zip(sort_idx[i: i + self.batch_size], tmp):
                    target_out_list.append((idx, record))
            target_out_list.sort()
            for idx, record in target_out_list:
                yield record
        else:
            for fasta_records_batch in self._batch(targets, self.batch_size):
                yield from self._batch_convert(fasta_records_batch)

    def _mask_and_get_probs(self, target_batch: List[str]) -> (Tuple[Tensor, List[List[int]]]):
        # prepare inputs:
        inputs = self.tokenizer(target_batch, return_tensors="pt", padding=True)  # [batch_size, L, vocab]
        if self.weight_type != torch.float32:
            for k, v in inputs.data.items():
                if k != 'input_ids': inputs.data[k] = v.to(self.weight_type)
        inputs.to(self.device)
        # apply mask:
        mask_positions: List[List[int]] = [[] for _ in range(len(target_batch))]
        for sequence_idx, sequence in enumerate(target_batch):
            for mask_idx in self._get_masked_positions(sequence):
                inputs["input_ids"][sequence_idx][mask_idx + 1] = self.tokenizer.mask_token_id # take into account [cls] at start
                mask_positions[sequence_idx].append(mask_idx)
        # run inference and return:
        with torch.no_grad():
            with torch.autocast("cuda"): 
                outputs = self.model(**inputs)
        probs: Tensor = torch.softmax(outputs.logits, dim=-1)  # [batch_size, L, vocab]
        probs = probs[:, 1:, :] # remove [cls]-entry
        return (probs, mask_positions)

    def _batch_convert(self, target_batch: List[str]) -> Iterator[str]:

        aa_ids = self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)

        k: int = 2 + len(self.special_amino_acids)  # why 2 - aa itself and I/L dillema

        probs, mask_positions = self._mask_and_get_probs(target_batch) # get the mask positions and probabilities from batch inference

        for sequence_idx, sequence in enumerate(target_batch):
            new_sequence: List[str] = list(sequence)
            for mask_position in mask_positions[sequence_idx]:
                top_idx: Tensor = None
                match self.ml_generator_type:
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