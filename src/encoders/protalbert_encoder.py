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
        output_list = self._embed_batched(sequences, batch_size = 1)  # get outputs of model for each batch
        embedding_list = [output.last_hidden_layer for output in output_list] # extract hidden layers from each output
        embeddings = torch.cat(embedding_list, dim=0) # Stack each of the batch hidden layers on top of each other, current dim is [Batch, max_tokenized_length, 1024]
        embeddings = embeddings.flatten(start_dim=1, end_dim=2) # Flatten last two dimensions, current dim is [Batch, max_tokenized_length * 1024]
        return embeddings
