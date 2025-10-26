import torch
import copy
import numpy as np

from src.peptide_classifiers.peptide_classifier import PeptideClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable

class MLPClassifier(PeptideClassifier, torch.nn.Module):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder):
        PeptideClassifier.__init__(self, encoder)
        torch.nn.Module.__init__(self)
        self.network = network

    def forward(self, x) -> torch.Tensor:
        return self.network(x)
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        x : torch.Tensor = self.encoder(sequences)
        x = self.forward(x)
        res = torch.round(x).tolist()
        return res
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> list[tuple[bool, float]]:
        x : torch.Tensor = self.encoder(sequences)
        x = self.forward(x)
        pred : torch.Tensor = torch.round(x)
        corr : torch.Tensor = (pred == outcomes)
        outcomes_tensor = torch.FloatTensor(list(outcomes))
        loss : torch.Tensor = torch.abs(x - outcomes_tensor)
        return list(zip(corr.tolist(), loss.tolist()))
    
def train_mlp(mlp : MLPClassifier, X_train : Iterable[str], y_train : Iterable[bool], X_val : Iterable[str], 
              y_val : Iterable[str], n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    #X_train = mlp.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).unsqueeze(1)
    #X_val = mlp.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).unsqueeze(1)

    loss_fn = torch.nn.BCELoss()

    best_acc = - np.inf
    best_weights = None
    for epoch in range(n_epochs):
        mlp.train()
        batch_starts = torch.arange(0, len(X_train), batch_size)
        for batch_start in batch_starts:
            batch_end = min(batch_start+batch_size, len(X_train))
            X_batch = mlp.encoder(X_train[batch_start:batch_end])
            y_batch = y_train[batch_start:batch_end]
            y_pred = mlp(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mlp.eval()
        batch_starts = torch.arange(0, len(X_val), batch_size)
        tot_acc = 0
        for batch_start in batch_starts:
            batch_end = min(batch_start+batch_size, len(X_val))
            X_batch = mlp.encoder(X_val[batch_start:batch_end])
            y_pred = mlp(X_batch)
            y_batch = y_val[batch_start:batch_end]
            acc = (y_pred.round() == y_batch).float().mean()
            tot_acc += (batch_end - batch_start) / len(X_val) * acc
        print("Validation accuracy after epoch %d: %.3f \n" % (epoch, tot_acc))
        if tot_acc > best_acc:
            best_acc = tot_acc
            best_weights = copy.deepcopy(mlp.state_dict())
        mlp.load_state_dict(best_weights) # restore best weights        