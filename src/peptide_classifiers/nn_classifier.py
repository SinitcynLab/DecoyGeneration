import torch
import copy
import numpy as np

from typing import Iterable, Union
from src.peptide_classifiers.peptide_classifier import PeptideClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from sklearn.model_selection import KFold

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
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        # get inputs:
        encodings : Union[torch.Tensor, list[torch.Tensor]] = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings)
        # cast outputs to list and return:
        return torch.round(out).tolist()
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> tuple[list[bool], list[float]]:
        # get inputs:
        encodings : Union[torch.Tensor, list[torch.Tensor]] = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings)
        # compute predictions, get accuracy:
        pred : torch.Tensor = torch.round(out)
        outcomes_tensor = torch.IntTensor(list(outcomes)).to(self.device)
        corr : torch.Tensor = torch.eq(pred, outcomes_tensor)
        loss : torch.Tensor = torch.abs(out - outcomes_tensor.float())
        # return whether predictions correct, and loss measure with each prediction:
        return corr.tolist(), loss.tolist()

def set_up_nn_training(nn : NNClassifier, X_train, y_train, X_val, y_val):
    X_train = nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).unsqueeze(1).to(nn.device)
    X_val = nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).unsqueeze(1).to(nn.device)

    loss_fn = torch.nn.BCELoss()

    best_acc = - np.inf
    best_weights = None

    return X_train, y_train, X_val, y_val, loss_fn, best_acc, best_weights

def cross_validate(nn: NNClassifier, sequences: Iterable[str], labels: Iterable[str], 
                   n_epochs: int, batch_size: int, 
                   optimizer: torch.optim.Optimizer, n_folds: int = 5) -> float:
    data: torch.Tensor = nn.encoder(sequences)
    labels: torch.Tensor = torch.FloatTensor(list(labels)).to(nn.device)

    loss_fn: torch.Module = torch.nn.BCELoss()
    mean_best_acc: int = 0

    kfold = KFold(n=n_folds, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data)):
        best_acc: float = - np.inf
        train_data: torch.Tensor = data[train_ids]
        val_data: torch.Tensor = data[val_ids]
        train_labels: torch.Tensor = labels[train_ids]
        val_labels: torch.Tensor = labels[val_ids]

        for epoch in range(n_epochs):
            _, val_acc = train_val_iteration(nn, train_data, train_labels, val_data, val_labels,
                                loss_fn, optimizer, batch_size)
            if val_acc > best_acc:
                best_acc = val_acc

        print(f"Best accuracy of {nn} on fold {fold}: {best_acc:.3f}.")
        mean_best_acc += best_acc / n_folds

    print(f"Mean best accuracy of {nn} on all {n_folds} folds: {mean_best_acc:.3f}.")
    return mean_best_acc

def train_val_iteration(nn: NNClassifier, train_data: torch.Tensor, val_data: torch.Tensor, 
                        train_labels: torch.Tensor, val_labels: torch.Tensor, loss: torch.Module, 
                        optimizer:torch.optim.Optimizer, batch_size: int):
    # train:
    N: int = len(train_data)
    train_acc: float = 0
    nn.train()
    batch_starts: torch.Tensor = torch.arange(0, N, batch_size)
    for batch_start in batch_starts:
        batch_end: int = min(batch_start + batch_size, N)
        batch_data: torch.Tensor = train_data[batch_start:batch_end]
        batch_labels: torch.Tensor = train_labels[batch_start:batch_end]
        acc = nn.train_on_batch(batch_data, batch_labels, loss, optimizer)
        train_acc += (batch_end - batch_start) / N * acc
    
    # validate:
    M: int = len(val_data)
    val_acc: float = 0
    nn.eval()
    batch_starts: torch.Tensor = torch.arange(0, M, batch_size)
    for batch_start in batch_starts:
        batch_end: int = min(batch_start + batch_size, M)
        batch_data: torch.Tensor = val_data[batch_start:batch_end]
        batch_labels: torch.Tensor = val_labels[batch_start:batch_end]
        acc = nn.evaluate_on_batch(batch_data, batch_labels)
        val_acc += (batch_end - batch_start) / M * acc

    return train_acc, val_acc

def train_nn(nn : NNClassifier, X_train : Iterable[str], y_train : Iterable[bool], X_val : Iterable[str], 
              y_val : Iterable[str], n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    X_train = nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).to(nn.device)
    X_val = nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).to(nn.device)
    
    loss_fn = torch.nn.BCELoss()

    best_val_acc = - np.inf
    best_weights = None

    N = len(X_train)
    M = len(X_val)
    for epoch in range(n_epochs):
        _, val_acc = train_val_iteration(nn, X_train, y_train, X_val, y_val, loss_fn, optimizer, batch_size)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(nn.state_dict())

    nn.load_state_dict(best_weights) # restore best weights
    return best_val_acc # return best validation accuracy