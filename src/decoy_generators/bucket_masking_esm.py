import torch
import math
import numpy as np
import copy

from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from src.decoy_generators.esm_generator import EsmGenerator
from typing import List, Tuple

from transformers.tokenization_utils import BatchEncoding
from random import Random
from torch import Tensor

class BucketMaskingEsmGenerator(EsmGenerator):
    def __init__(
            self,
            local_path: str,
            random: Random,
            special_amino_acids: List[str],
            sort_optimization: bool = True,
            batch_size: int = 64,
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST,
            device: torch.device = 'cpu',
            weight_type: torch.dtype = torch.float32,
            masking_count: int = 1,
            bucket_count: int = 10,
            sample_count: int = 3
    ):
        EsmGenerator.__init__(self, local_path, random, special_amino_acids, sort_optimization,
                             batch_size, ml_generator_type, device, MaskingType.COUNT, 0, masking_count, weight_type)
        self.aa_ids = self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        self.bucket_count = bucket_count
        self.sample_count = sample_count

    def _mask_and_get_probs(self, target_batch: List[str]) -> (Tuple[Tensor, List[List[int]]]):
        inputs = self.tokenizer(target_batch, return_tensors="pt", padding=True)  # [batch_size, L, vocab]
        if self.weight_type != torch.float32:
            for k, v in inputs.data.items():
                if k != 'input_ids': inputs.data[k] = v.to(self.weight_type)
        inputs.to(self.device)

        sample_list: List[Tensor] = []
        # collect each 'sample', representing the probability results for all AAs resulting from a specific binning:
        for sample_id in range(self.sample_count):
            sample_list.append(self._get_sample(target_batch, inputs))
        sample_mean = torch.mean(torch.stack(sample_list), dim = 0) # get mean of all samples to select 'optimal' places

        special_indices = [i for i in range(len(self.canonical_amino_acids)) if self.canonical_amino_acids[i] in self.special_amino_acids]
        sequence_means: List[Tensor] = []
        sequence_masks: List[List[int]] = []
        for seq_id, sequence in enumerate(target_batch):
            sequence_mask: List[int] = []
            L = len(sequence)
            # maps position 'i' to the index of the amino acid orignally present at 'i':
            disable_dict: dict = {i: self.canonical_amino_acids.index(sequence[i]) for i in range(L)}
            sequence_mean = sample_mean[seq_id, :, :].squeeze(0) # [L, vocab_size]
            sequence_means.append(sequence_mean)
            for i in range(L): # for each position
                sequence_mean[i, [disable_dict[i]] + special_indices] = 0 # set the probability of the originally present and special amino acids very low
            for peptide in self.get_all_peptides(sequence):
                peptide_mean = sequence_mean[peptide[0]:peptide[-1]+1, :] # [peptide_length, vocab_size]
                peptide_max = peptide_mean.max(dim=1).values # [peptide_length]
                k = min(self.mask_count, len(peptide))
                _, optimal_pos = torch.topk(peptide_max, k=k, largest=True) # get positions with highest max prob
                sequence_mask = sequence_mask + [i for i in optimal_pos] # append them to the mask for this sequence
            sequence_masks.append(sequence_mask) # append mask of this sequence to list of all masks

        output_probs = torch.stack(sequence_means)

        return (output_probs, sequence_masks)
    
    def _get_sample(self, target_batch: List[str], inputs: BatchEncoding) -> List[List[float]]:
        L = max([len(seq) for seq in target_batch])
        vocab_size = self.tokenizer.vocab_size
        batch_size = len(target_batch)
        sample_probs = torch.zeros((batch_size, L, vocab_size))

        bucket_list: List[List[List[int]]] = [self._generate_buckets(seq, self.bucket_count) for seq in target_batch] # create buckets
        for bucket_id in range(self.bucket_count): # for each bucket...
            input_copy = copy.copy(inputs) # create fresh input
            for seq_id in range(batch_size): # for each sequence...
                input_copy["input_ids"][seq_id][bucket_list[seq_id][bucket_id]] = self.tokenizer.mask_token_id # mask all AAs in current bucket
            # Calculate distributions for all AAs in the current bucket in each sequence, using ONE call to model:
            with torch.no_grad():
                with torch.autocast("cuda"):
                    outputs = self.model(**input_copy)
            bucket_probs = torch.softmax(outputs.logits, dim=-1).cpu()
            # for each sequence, save probs in this bucket:
            for seq_id in range(batch_size):
                current_bucket = torch.tensor(bucket_list[seq_id][bucket_id], dtype=int)
                current_bucket_incr = current_bucket + 1 # add 1 because first token is now CLS token (final token is SEP token)
                sample_probs[seq_id, current_bucket, :] = bucket_probs[seq_id][current_bucket_incr][:]
                # TODO: double-check if above line sets correctly by debugging.

        return sample_probs # return collected probs

    def _generate_buckets(self, sequence: str, num_buckets: int) -> List[List[int]]:
        sequence_length = len(sequence)
        entries = list(range(sequence_length))
        Random.shuffle(self.random, entries) # note: shuffle works in place
        bucket_size = math.ceil(sequence_length / num_buckets)
        buckets = [entries[i*bucket_size:(i+1)*bucket_size] for i in range(num_buckets)]
        return buckets

    def __str__(self):
        esm_string = EsmGenerator.__str__(self)
        return ("bucket_masking_" + esm_string)