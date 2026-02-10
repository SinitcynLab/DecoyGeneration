import torch

from transformers import EsmTokenizer, EsmForMaskedLM

from src.encoders.transformer_encoder import TransformerEncoder
from src.cli.option_lists import PARAM_COUNT_TO_PATH

class EsmEncoder(TransformerEncoder):
    def __init__(self, max_tokenized_length : int = 256, device : torch.device = 'cpu', 
                 constant_length : bool = True, flatten : bool = True, parameter_count: str = "8M"):
        TransformerEncoder.__init__(self, max_tokenized_length, device, constant_length, flatten)
        local_path = PARAM_COUNT_TO_PATH[parameter_count]
        self.model = EsmForMaskedLM.from_pretrained(local_path, local_files_only=True)
        self.tokenizer = EsmTokenizer.from_pretrained(local_path, local_files_only=True)
        self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        self.flatten = flatten
        self.model.eval()
        self.model.to(self.device)