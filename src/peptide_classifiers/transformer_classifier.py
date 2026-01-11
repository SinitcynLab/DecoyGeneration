import torch

from typing import Callable

from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.peptide_encoder import PeptideEncoder

class TransformerClassifier(FeedForwardNNClassifier):
    def __init__(self, network: torch.nn.Sequential, embedding: torch.nn.Linear, pos_embedding: torch.nn.Embedding, 
                 transformer: torch.nn.TransformerEncoder, encoder: PeptideEncoder, name: str, device: torch.device, resetter: Callable = None):
        FeedForwardNNClassifier.__init__(self, network, encoder, name, device)
        self.embedding = embedding
        self.pos_embedding = pos_embedding
        self.transformer = transformer
        self.resetter = resetter

    def forward(self, x):
        x = x.unsqueeze(-1)
        batch_size, seq_len, _ = x.shape

        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        x = self.embedding(x) + self.pos_embedding(positions)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.network(x).squeeze(-1)
    
    def set_device(self, device):
        self.embedding.to(device)
        self.pos_embedding.to(device)
        self.transformer.to(device)
        FeedForwardNNClassifier.set_device(self, device)

    def reset(self):
        if self.resetter == None:
            raise ValueError("Net setter must be set in order to reset NN classifier.")
        else:
            self.network, self.embedding, self.pos_embedding, self.transformer = self.resetter()
            self.set_device(self.device)