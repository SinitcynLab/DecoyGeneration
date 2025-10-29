import torch

from transformers import EsmTokenizer, EsmForMaskedLM
from typing import Iterable

from src.encoders.transformer_encoder import TransformerEncoder

class EsmEncoder(TransformerEncoder):
    def __init__(self, max_tokenized_length : int = 256, device : torch.device = 'cpu'):
        TransformerEncoder.__init__(self, max_tokenized_length, device)
        LOCAL_PATH = "models/esm2_t6_8M_UR50D"
        self.model = EsmForMaskedLM.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.tokenizer = EsmTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, sequences : Iterable[str]):
        output_list = self._embed_batched(sequences) # [n.o. sequences, max_tokenized_length, 320]
        embeddings = embeddings.flatten(start_dim = 1, end_dim = 2) # [n.o. sequences, max_tokenized_length * 320]
        return embeddings