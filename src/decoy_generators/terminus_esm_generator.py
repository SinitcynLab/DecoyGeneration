
import torch

from src.decoy_generators.esm_generator import EsmGenerator
from src.decoy_generators.ml_generator import MlGeneratorType, MaskingType
from typing import List

from random import Random

torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TerminusEsmGenerator(EsmGenerator):
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
            weight_type: torch.dtype = torch.float32,
            terminus: str = 'C'
    ):
        EsmGenerator.__init__(self, local_path, random, special_amino_acids, sort_optimization,
                             batch_size, ml_generator_type, device, masking_type, mask_percent, mask_count, weight_type)
        if terminus not in ['N', 'C']:
            raise ValueError("Please provide valid terminus")
        self.terminus = terminus

    def __str__(self):
        out = EsmGenerator.__str__(self)
        out = self.terminus + "_terminus_" + out
        return out
    
    def _select_mask(self, start: int, end: int, size: int):
        if self.terminus == 'N':
            return [start]
        elif self.terminus == 'C':
            return [end - 1]