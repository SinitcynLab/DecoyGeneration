import torch

from src.encoders.transformer_encoder import TransformerEncoder
from typing import Iterable
from transformers import BertModel, BertTokenizer

class ProtBertEncoder(TransformerEncoder):
    def __init__(self, max_tokenized_length : int = 64, device='cpu', constant_length : bool = True, flatten : bool = True):
        TransformerEncoder.__init__(self, max_tokenized_length, device, constant_length)
        LOCAL_PATH = "models/prot_bert"
        self.model = BertModel.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        self.model.eval()
        self.model.to(self.device)
        self.flatten = flatten

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        sequences = [" ".join(sequence) for sequence in sequences]
        if self.constant_length:
            embeddings = self._embed_batched_constant_length(sequences, batch_size = 1) # [Batch, max_tokenized_length, 1024]
        else:
            embeddings = self._embed_batched_varied_length(sequences)
        if self.flatten:
            embeddings = embeddings.flatten(start_dim=1, end_dim=2) # [Batch, max_tokenized_length * 1024]
        return embeddings