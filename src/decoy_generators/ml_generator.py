import math
import torch
import numpy as np

from typing import Iterator, List, Tuple
from random import Random
from enum import Enum
from torch import Tensor
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from src.decoy_generators.decoy_generator import DecoyGenerator

from src.proteins.protease import Protease

from collections import Counter


class MaskingType(Enum):
    COUNT = 1,
    PERCENT = 2
    N_C_TERMINUS = 4

class MlGeneratorType(Enum):
    BEST = 1,
    WORST = 2

class MlGenerator(DecoyGenerator):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        model_name: str,
        random: Random,
        protease: Protease,
        sort_optimization: bool = True,
        batch_size: int = 64,
        ml_generator_type: MlGeneratorType = MlGeneratorType.BEST,
        device : torch.device = 'cpu',
        masking_type: MaskingType = MaskingType.PERCENT,
        mask_percent: float = 0.3,
        mask_count: int = 1,
        dtype: torch.dtype = torch.float32
    ):
        DecoyGenerator.__init__(self, protease)
    
        self.model_name = model_name
        self.random = random
        self.sort_optimization = sort_optimization
        self.batch_size = batch_size
        self.ml_generator_type = ml_generator_type
        self.device = device
        self.masking_type = masking_type
        self.mask_percent = mask_percent
        self.mask_count = mask_count
        self.dtype = dtype

        self._c_replacements = {}
        self._n_replacements = {}
        self._n_c_replacements = {}

    def _get_masked_positions(self, sequence: str):
        peptide_start_idx = 0
        for peptide in self.protease.cleave(sequence):
            peptide_length = len(peptide.sequence)
            peptide_end_idx = peptide_start_idx + peptide_length

            first_can_change = peptide.flexible_range[0] if len(peptide.flexible_range) > 0 else None
            last_can_change = peptide.flexible_range[-1] if len(peptide.flexible_range) > 0 else None

            if self.masking_type == MaskingType.N_C_TERMINUS:
                if len(peptide.flexible_range) > 1:
                    yield first_can_change + peptide_start_idx  # N-terminus
                    yield last_can_change + peptide_start_idx  # C-terminus
                peptide_start_idx = peptide_end_idx
                continue

            mask_count = self._get_masking_count(peptide_length)
            if mask_count > 0:
                # TODO(Grigory, Fabrice): for now we mutate only uncontrained aminoacids, but
                # we can also mutate constraned ones as long as we do not violate the cleavage constraints.
                # For example, we can change K to R in a C terminus of a tryptic peptide
                for peptide_mask_idx in self.random.sample(peptide.flexible_range, min(mask_count, len(peptide.flexible_range))):
                    # Switch from peptide-relative to sequence-relative indexing
                    yield peptide_start_idx + peptide_mask_idx

            # Switch to next peptide
            peptide_start_idx = peptide_end_idx

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
                    _ for _ in self._convert_batch([target_list[idx] for idx in sort_idx[i: i + self.batch_size]])
                ]
                for idx, record in zip(sort_idx[i: i + self.batch_size], tmp):
                    target_out_list.append((idx, record))
            target_out_list.sort()
            for idx, record in target_out_list:
                yield record
        else:
            for fasta_records_batch in self._batch(targets, self.batch_size):
                yield from self._convert_batch(fasta_records_batch)

    def _prepare_inputs(self, target_batch: List[str]) -> dict:
        inputs = self.tokenizer(target_batch, return_tensors="pt", padding=True)  # [batch_size, L, vocab]
        for k, v in inputs.data.items():
            if k != 'input_ids':
                inputs.data[k] = v.to(self.dtype)
        inputs.to(self.device)
        return inputs

    def _mask_and_get_probs(self, target_batch: List[str]) -> (Tuple[Tensor, List[List[int]]]):
        # prepare inputs:
        inputs = self._prepare_inputs(target_batch)
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

    def _convert_batch(self, batch: List[str]) -> Iterator[str]:
        aa_ids = self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)

        probs, mask_positions = self._mask_and_get_probs(batch) # get the mask positions and probabilities from batch inference

        for sequence_idx, sequence in enumerate(batch):
            new_sequence: List[str] = list(sequence)
            peptides = self.protease.cleave(sequence)
            allowed_replacements = [replacements for peptide in peptides for replacements in peptide.allowed_replacements]

            for mask_position in mask_positions[sequence_idx]:
                top_idx: Tensor = None
                match self.ml_generator_type:
                    case MlGeneratorType.BEST:
                        _, top_idx = torch.topk(probs[sequence_idx, mask_position, aa_ids], k=len(aa_ids), largest=True)
                    case MlGeneratorType.WORST:
                        _, top_idx = torch.topk(probs[sequence_idx, mask_position, aa_ids], k=len(aa_ids), largest=False)

                original_aa: str = sequence[mask_position]

                for idx in top_idx:
                    new_aa: str = self.canonical_amino_acids[idx]
                    # We need to change something
                    if new_aa == original_aa:
                        continue
                    # We should respect protease cleavage rules
                    if new_aa not in allowed_replacements[mask_position]:
                        continue
                    # I<->L mutations are not allowed since they are indistinguishable in MS
                    if (new_aa == 'I' and original_aa == 'L') or (
                            new_aa == 'L' and original_aa == 'I'):
                        continue
                    new_sequence[mask_position] = new_aa
                    #self._log_data(probs, sequence_idx, mask_position, sequence, new_aa, self.canonical_amino_acids[top_idx[0]]) # for visualization
                    break

            # Always report the same decoy for the same peptide, even across different proteins.
            for peptide in peptides:
                old_peptide = peptide.sequence
                new_peptide = "".join(new_sequence[peptide.start_index:peptide.end_index])
                if old_peptide in self.peptide_cache:
                    new_peptide = self.peptide_cache[old_peptide][0]
                else:
                    self.peptide_cache[old_peptide] = [new_peptide]

                for i in range(peptide.start_index, peptide.end_index):
                    new_sequence[i] = new_peptide[i - peptide.start_index]
                
            yield "".join(new_sequence)

    def _log_data(self, probs: Tensor, sequence_idx: int, mask_position: int, sequence: str, chosen_aa: str, top_aa: str):
        aa_i = sequence[mask_position]
        aa_i_min_1 = sequence[mask_position - 1]
        if mask_position + 1 < len(sequence):
            aa_i_plus_1 = sequence[mask_position + 1]
        else:
            aa_i_plus_1 = ""
        relevant_aa_ids = self.tokenizer.convert_tokens_to_ids([aa_i, chosen_aa, top_aa])
        relevant_aa_probs = probs[sequence_idx, mask_position, relevant_aa_ids] # [prob_og_aa, prob_chosen_aa, prob_top_aa]
        # log the original token:
        with open(f'og_aa_{self}.txt', 'a') as file:
            file.write(f"{aa_i}\n")
        # log the offset tokens:
        with open(f'aa_offset_{self}.txt', 'a') as file:
            file.write(f"{aa_i_min_1},{aa_i_plus_1}\n")
        # log the most-probable token:
        with open(f'most_probable_aa_{self}.txt', 'a') as file:
            file.write(f"{top_aa}\n")
        # log the chosen token:
        with open(f'token_choices_{self}.txt', 'a') as file:
            file.write(f"{chosen_aa}\n")
        # log associated probabilities:
        save_array = relevant_aa_probs.cpu().numpy()
        with open(f'prob_distr_{self}.txt', 'a') as file:
            np.savetxt(file, save_array)
            file.write("\n")

    def __str__(self) -> str:
        name = self.model_name.replace("/", "_")

        if self.masking_type == MaskingType.PERCENT:
            mask_percent: str = f"{self.mask_percent}".replace(".", "") # avoid also using '.' for decimal point
            name += f".{self.ml_generator_type.name.lower()}.p{mask_percent}"
        elif self.masking_type == MaskingType.COUNT:
            name += f".{self.ml_generator_type.name.lower()}.c{self.mask_count}"
        elif self.masking_type == MaskingType.N_C_TERMINUS:
            name += f".{self.ml_generator_type.name.lower()}.n_c_term"
        else:
            raise ValueError(f"Unsupported masking type: {self.masking_type}")

        if self.dtype == torch.float16:
            name += ".f16"
        elif self.dtype == torch.float32:
            name += ".f32"
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        return name
