import torch
import copy
import numpy as np

from typing import Iterable
from src.peptide_classifiers.peptide_classifier import PeptideClassifier
from src.encoders.peptide_encoder import PeptideEncoder

class NNClassifier(PeptideClassifier, torch.nn.Module):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder, device : torch.device):
        PeptideClassifier.__init__(self, encoder, device)
        torch.nn.Module.__init__(self)
        self.network = network
        network.to(device)

    def set_device(self, device : torch.device):
        PeptideClassifier.set_device(device)
        self.network.to(device)

    def evaluate_on_batch(self, X_batch, y_batch) -> tuple[torch.Tensor, float]:
        y_pred = self(X_batch)
        acc = (y_pred.round() == y_batch).float().mean()
        return y_pred, acc

    def train_on_batch(self, X_batch, y_batch, loss_fn, optimizer) -> float:
        y_pred, acc = self.evaluate_on_batch(X_batch, y_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return acc

def set_up_nn_training(nn : NNClassifier, X_train, y_train, X_val, y_val):
    X_train = nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).unsqueeze(1).to(nn.device)
    X_val = nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).unsqueeze(1).to(nn.device)

    loss_fn = torch.nn.BCELoss()

    best_acc = - np.inf
    best_weights = None

    return X_train, y_train, X_val, y_val, loss_fn, best_acc, best_weights

def train_nn(nn : NNClassifier, X_train : Iterable[str], y_train : Iterable[bool], X_val : Iterable[str], 
              y_val : Iterable[str], n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    X_train = nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).to(nn.device)
    X_val = nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).to(nn.device)
    
    loss_fn = torch.nn.BCELoss()

    best_acc = - np.inf
    best_weights = None

    N = len(X_train)
    M = len(X_val)
    for epoch in range(n_epochs):
        nn.train()
        batch_starts = torch.arange(0, N, batch_size)
        tot_acc = 0
        for batch_start in batch_starts:
            batch_end = min(batch_start+batch_size, N)
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            acc = nn.train_on_batch(X_batch, y_batch, loss_fn, optimizer)
            tot_acc += (batch_end - batch_start) / N * acc
        nn.eval()
        batch_starts = torch.arange(0, M, batch_size)
        tot_acc = 0
        for batch_start in batch_starts:
            batch_end = min(batch_start+batch_size, M)
            X_batch = X_val[batch_start:batch_end]
            y_batch = y_val[batch_start:batch_end]
            _, acc = nn.evaluate_on_batch(X_batch, y_batch)
            tot_acc += (batch_end - batch_start) / M * acc
        if tot_acc > best_acc:
            best_acc = tot_acc
            best_weights = copy.deepcopy(nn.state_dict())
    nn.load_state_dict(best_weights) # restore best weights
    return best_acc  