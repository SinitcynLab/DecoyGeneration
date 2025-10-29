import torch
import re

from transformers import AlbertModel, AlbertTokenizer
from typing import Iterable

from src.encoders.transformer_encoder import TransformerEncoder

class ProtAlbertEncoder(TransformerEncoder):
    def __init__(self, max_tokenized_length : int = 256, device : torch.device = 'cpu'):
        TransformerEncoder.__init__(self, max_tokenized_length, device)
        LOCAL_PATH = "Rostlab/prot_albert"
        self.model = AlbertModel.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.tokenizer = AlbertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
        self.model.eval()
        self.model.to(self.device)
        print("initialized")

    def __call__(self, sequences : Iterable[str]) -> torch.Tensor:
        sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences] # Remove unwanted characters
        sequences = [" ".join(sequence) for sequence in sequences] # Add white spaces between characters
        embeddings = self._embed_batched(sequences, batch_size=1) # [n.o. sequences, max_tokenized_length, 4096]
        print(embeddings.shape)
        embeddings = embeddings.flatten(start_dim = 1, end_dim = 2) # [n.o. sequences, max_tokenized_length * 4096]
        return embeddings
