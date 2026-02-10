
import torch

from src.decoy_generators.decoy_generator import DecoyGeneratorType
from src.decoy_generators.ml_generator import MlGenerator, MaskingType, MlGeneratorType
from typing import List
from transformers import BertModel, BertTokenizer

from random import Random


class ProtBertGenerator(MlGenerator):
    model: BertModel
    tokenize: BertTokenizer
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
            weight_type: torch.dtype = torch.float32
    ):
        MlGenerator.__init__(self, local_path, random, special_amino_acids, sort_optimization,
                             batch_size, ml_generator_type, device, masking_type, mask_percent, mask_count, weight_type)
        self.model = BertModel.from_pretrained(local_path, local_files_only=True, torch_dtype=weight_type)
        self.tokenizer = BertTokenizer.from_pretrained(local_path, local_files_only=True, torch_dtype=weight_type)
        self.model.eval()
        self.model.to(self.device)

    def __str__(self):
        out: str = "protbert"

        if self.masking_type == MaskingType.PERCENT:
            mask_percent: str = f"{self.mask_percent}".replace(".", "") # avoid also using '.' for decimal point
            out = out + f".{self.ml_generator_type.name.lower()}.p{mask_percent}"
        elif self.masking_type == MaskingType.COUNT:
            out = out + f".{self.ml_generator_type.name.lower()}.c{self.mask_count}"

        if self.weight_type == torch.float16:
            out = out + ".16b"
        
        return out
