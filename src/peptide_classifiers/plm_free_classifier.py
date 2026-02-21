import torch

import torch.nn as nn
from typing import Iterable

from src.peptide_classifiers.recurrent_nn_classifier import RecurrentNNClassifier
from src.encoders.peptide_encoder import PeptideEncoder

class PlmFreeClassifier(RecurrentNNClassifier):
    def __init__(
        self,
        embedding: nn.Embedding,
        rnn: nn.RNN,
        network: nn.Sequential,
        encoder: PeptideEncoder,
        name: str = "plm_free_classifier",
        device: str = "cpu", 
        resetter = None
    ):
        super().__init__(rnn, network, encoder, name, device, resetter)
        self.embedding = embedding.to(self.device)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        # h_n: (num_layers * num_directions, batch, hidden)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        logits = self.network(h).squeeze(-1)
        return torch.sigmoid(logits)
    
    def collate_pad(self, tensor_list: Iterable[torch.Tensor], pad_id: int):
        lengths: torch.Tensor = torch.tensor([len (t) for t in tensor_list], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(lengths) else 0
        out: torch.Tensor = torch.full((len(tensor_list), max_len), pad_id, dtype=torch.long)
        for i, t in enumerate(tensor_list):
            out[i,:len(t)] = torch.tensor(t, dtype=torch.long)
        return out, lengths
    
    def evaluate_on_data(self, tensor_list: Iterable[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            X, l = self.collate_pad(tensor_list, self.encoder.pad_id)
            X, l = X.to(self.device), l.to(self.device)
            y_pred = self(X, l)
            del X, l
            torch.cuda.empty_cache()
            return y_pred
    
    def train_on_data(self, tensor_list: Iterable[torch.Tensor], y: torch.Tensor, 
                    loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        X, l = self.collate_pad(tensor_list, self.encoder.pad_id)
        X, l, y = X.to(self.device), l.to(self.device), y.to(self.device)
        y_pred = self(X, l)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del X, l, y, loss
        torch.cuda.empty_cache()
        return y_pred
    
    def set_device(self, device):
        RecurrentNNClassifier.set_device(self, device)
        self.embedding.to(device)
    
    def reset(self):
        net, rnn, embedding = self.resetter()
        self.network = net
        self.rnn = rnn
        self.embedding = embedding
        self.set_device(self.device)