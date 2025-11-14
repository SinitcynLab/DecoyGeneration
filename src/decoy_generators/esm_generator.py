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
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST,
            device: torch.device = 'cpu'
    ):
        MlGenerator.__init__(self, local_path, random, special_amino_acids, mask_percent, sort_optimization,
                             batch_size, ml_generator_type, device)
        self.model = EsmForMaskedLM.from_pretrained(local_path, local_files_only=True)
        self.tokenizer = EsmTokenizer.from_pretrained(local_path, local_files_only=True)
        self.model.eval()
        self.model.to(self.device)

    def __str__(self):
        param_count = self.local_path.split('/')[-1].split('_')[2]
        return f"esm{param_count}.{self.esm_generator_type.name.lower()}.[{self.mask_percent}]"
