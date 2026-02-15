import torch

from src.encoders.peptide_encoder import PeptideEncoder
from torch_pca import PCA
from typing import Iterable

class PcaEncoder(PeptideEncoder):
    def __init__(self, wrapped_encoder : PeptideEncoder, pca_dim : int, device : torch.device):
        PeptideEncoder.__init__(self)
        self.pca = PCA(n_components=pca_dim)
        self.has_been_fit = False
        self.wrapped_encoder = wrapped_encoder
        self.device = device
    
    def fit_pca(self, embeddings: torch.Tensor):
        self.pca.fit(embeddings)
        self.has_been_fit = True

    def __call__(self, sequences : Iterable[str]):
        embeddings = self.wrapped_encoder(sequences).to(self.device)
        # PCA is automatically fit to the first sequence iterable you execute it on:
        if not (self.has_been_fit):
            X = self.fit_pca(embeddings) # PCA package expects and returns transpose
            X = self.pca.transform(embeddings)
        else:
            X = self.pca.transform(embeddings)
        return X
        