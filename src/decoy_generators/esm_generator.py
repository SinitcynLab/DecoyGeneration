
import torch

from src.decoy_generators.decoy_generator import DecoyGeneratorType
from src.decoy_generators.ml_generator import MlGenerator, MaskingType, MlGeneratorType
from typing import List
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
    ml_generator_type: MlGeneratorType

    decoy_generation_type: DecoyGeneratorType = DecoyGeneratorType.ONE2MANY

    def __init__(
            self,
            local_path: str,
            random: Random,
            special_amino_acids: List[str],
            sort_optimization: bool = True,
            batch_size: int = 64,
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST,
            device: torch.device = 'cpu',
            masking_type: MaskingType = MaskingType.PERCENT,
            mask_percent: float = 0.3,
            mask_count: int = 1,
            weight_type: torch.dtype = torch.float32 # source: https://huggingface.co/blog/accelerate-large-models
    ):
        MlGenerator.__init__(self, local_path, random, special_amino_acids, sort_optimization,
                             batch_size, ml_generator_type, device, masking_type, mask_percent, mask_count)
        self.model = EsmForMaskedLM.from_pretrained(local_path, local_files_only=True, torch_dtype=weight_type)
        self.tokenizer = EsmTokenizer.from_pretrained(local_path, local_files_only=True, torch_dtype=weight_type)
        self.model.eval()
        self.model.to(self.device)

    def __str__(self):
        param_count = self.local_path.split('/')[-1].split('_')[2]
        if self.masking_type == MaskingType.PERCENT:
            mask_percent = f"{self.mask_percent}".replace(".", "_") # avoid also using '.' for decimal point
            return f"esm{param_count}.{self.ml_generator_type.name.lower()}.p{self.mask_percent}"
        elif self.masking_type == MaskingType.COUNT:
            return f"esm{param_count}.{self.ml_generator_type.name.lower()}.c{self.mask_count}"
