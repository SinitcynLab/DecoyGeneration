import torch

from transformers import EsmTokenizer, EsmForMaskedLM

from src.encoders.transformer_encoder import TransformerEncoder
from src.cli.option_lists import get_model_name

class EsmEncoder(TransformerEncoder):
    def __init__(
        self,
        max_tokenized_length: int = 256,
        device: torch.device = 'cpu', 
        constant_length: bool = True,
        flatten: bool = True,
        parameter_count: str = "650M",
        dtype: torch.dtype = torch.float32,
    ):
        TransformerEncoder.__init__(self, max_tokenized_length, device, constant_length, flatten)

        model_name = get_model_name(model_type="esm", model_size=parameter_count)

        self.model = EsmForMaskedLM.from_pretrained(model_name, dtype=dtype)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        self.flatten = flatten
        self.model.eval()
        self.model.to(self.device)
    