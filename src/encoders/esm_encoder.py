import torch

from transformers import EsmTokenizer, EsmForMaskedLM

from src.encoders.transformer_encoder import TransformerEncoder

class EsmEncoder(TransformerEncoder):
    def __init__(self, max_tokenized_length : int = 256, device : torch.device = 'cpu', constant_length : bool = True, flatten : bool = True):
        TransformerEncoder.__init__(self, max_tokenized_length, device, constant_length, flatten)
        LOCAL_PATH = "models/esm2_t6_8M_UR50D"
        self.model = EsmForMaskedLM.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.tokenizer = EsmTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        self.flatten = flatten
        self.model.eval()
        self.model.to(self.device)