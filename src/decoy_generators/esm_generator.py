
import torch

from src.decoy_generators.decoy_generator import DecoyGeneratorType
from src.decoy_generators.ml_generator import MlGenerator, MaskingType, MlGeneratorType

from src.proteins.protease import Protease

from typing import List
from transformers import EsmTokenizer, EsmForMaskedLM

from random import Random

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
        model_name: str,
        random: Random,
        protease: Protease,
        sort_optimization: bool = True,
        batch_size: int = 64,
        ml_generator_type: MlGeneratorType = MlGeneratorType.BEST,
        device: torch.device = 'cpu',
        masking_type: MaskingType = MaskingType.PERCENT,
        mask_percent: float = 0.3,
        mask_count: int = 1,
        dtype: torch.dtype = torch.float32 # source: https://huggingface.co/blog/accelerate-large-models
    ):
        MlGenerator.__init__(self, model_name, random, protease, sort_optimization,
                             batch_size, ml_generator_type, device, masking_type, mask_percent, mask_count, dtype)

        self.model = EsmForMaskedLM.from_pretrained(model_name, torch_dtype=dtype)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        self.model.to(self.device)
