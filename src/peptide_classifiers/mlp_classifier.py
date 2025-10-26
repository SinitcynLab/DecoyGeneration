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
    X_train = mlp.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).unsqueeze(1)
    X_val = mlp.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).unsqueeze(1)

    loss_fn = torch.nn.BCELoss()
    batch_starts = torch.arange(0, len(X_train), batch_size)

    best_acc = - np.inf
    best_weights = None
    print(X_train.shape)
    for epoch in range(n_epochs):
        mlp.train()
        for batch_start in batch_starts:
            batch_end = min(batch_start+batch_size, X_train.shape[0])
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            y_pred = mlp(X_batch)
            loss = loss_fn(y_pred, y_batch)
            acc = (y_pred.round() == y_batch).float().mean()
            print("Batch acc: %.2f \n" % float(acc))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mlp.eval()
        y_pred = mlp(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        print("Validation accuracy after epoch %d: %.3f \n" % (epoch, acc))
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(mlp.state_dict())
        mlp.load_state_dict(best_weights) # restore best weights        