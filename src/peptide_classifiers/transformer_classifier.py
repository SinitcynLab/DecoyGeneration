import torch

from typing import Callable

from src.peptide_classifiers.recurrent_nn_classifier import RecurrentNNClassifier
from src.encoders.peptide_encoder import PeptideEncoder

class TransformerClassifier(RecurrentNNClassifier):
    def __init__(self, network: torch.nn.Sequential, embedding: torch.nn.Linear, pos_embedding: torch.nn.Embedding, 
                 transformer: torch.nn.TransformerEncoder, encoder: PeptideEncoder, name: str, device: torch.device, resetter: Callable = None):
        RecurrentNNClassifier.__init__(self, torch.nn.RNN(input_size=1, hidden_size=1), network, encoder, name, device)
        self.embedding = embedding
        self.pos_embedding = pos_embedding
        self.transformer = transformer
        self.resetter = resetter
        self.set_device(self.device)

    def forward(self, x, l):
        out = torch.zeros(x.shape[0]).to(self.device)
        for i, t in enumerate(x):
            t = t[0:l[i],:] # keep only first l[i] timesteps
            positions = torch.arange(l[i], device=self.device).unsqueeze(0)
            t = self.embedding(t) + self.pos_embedding(positions)

            t = self.transformer(t)

            t = t.mean(dim=1)

            out[i] = self.network(t)

        return out
    
    def set_device(self, device):
        self.embedding.to(device)
        self.pos_embedding.to(device)
        self.transformer.to(device)
        RecurrentNNClassifier.set_device(self, device)

    def reset(self):
        if self.resetter == None:
            raise ValueError("Net setter must be set in order to reset NN classifier.")
        else:
            self.network, self.embedding, self.pos_embedding, self.transformer = self.resetter()
            self.set_device(self.device)