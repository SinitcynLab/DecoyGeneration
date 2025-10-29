import torch

from src.encoders.transformer_encoder import TransformerEncoder
from typing import Iterable
from transformers import BertModel, BertTokenizer

class ProtBertEncoder(TransformerEncoder):
    def __init__(self, max_tokenized_length : int = 64, device='cpu'):
        TransformerEncoder.__init__(self, max_tokenized_length, device)
        LOCAL_PATH = "models/prot_bert"
        self.model = BertModel.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        sequences = [" ".join(sequence) for sequence in sequences]
        embeddings = self._embed_batched(sequences, batch_size = 1) # [Batch, max_tokenized_length, 1024]
        embeddings = embeddings.flatten(start_dim=1, end_dim=2) # [Batch, max_tokenized_length * 1024]
        return embeddings