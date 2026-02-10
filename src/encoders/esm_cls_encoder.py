import torch

from src.encoders.esm_encoder import EsmEncoder
from typing import Iterable

class EsmClsEncoder(EsmEncoder):
    def __init__(self, device='cpu', parameter_count: str = "650M"):
        EsmEncoder.__init__(self, device=device, constant_length=False, parameter_count=parameter_count)
        self.cls_only = True