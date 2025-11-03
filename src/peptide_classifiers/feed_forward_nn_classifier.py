import torch
import copy
import numpy as np

from src.peptide_classifiers.nn_classifier import NNClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable

class FeedForwardNNClassifier(NNClassifier, torch.nn.Module):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder, device : torch.device):
        NNClassifier.__init__(self, network, encoder, device)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.network(x))
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        x : torch.Tensor = self.encoder(sequences)
        x = self.forward(x)
        res = torch.round(x).tolist()
        return res
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> tuple[list[bool], list[float]]:
        x : torch.Tensor = self.encoder(sequences)
        x = self.forward(x)
        pred : torch.Tensor = torch.round(x)
        outcomes_tensor = torch.IntTensor(list(outcomes)).to(self.device)
        corr : torch.Tensor = torch.eq(pred, outcomes_tensor)
        print(x.device)
        loss : torch.Tensor = torch.abs(x - outcomes_tensor.float())
        return corr.tolist(), loss.tolist()
    
    def set_device(self, device : torch.device):
        NNClassifier.set_device(device)
    
def train_feed_forward_nn(feed_foward_nn : FeedForwardNNClassifier, X_train : Iterable[str], y_train : Iterable[bool], X_val : Iterable[str], 
              y_val : Iterable[str], n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    X_train = feed_foward_nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).to(feed_foward_nn.device)
    X_val = feed_foward_nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).to(feed_foward_nn.device)

    loss_fn = torch.nn.BCELoss()

    best_acc = - np.inf
    best_weights = None

    print("Pre-training accuracy on training set: %.2f" % (feed_foward_nn(X_train) == y_train).float().mean())
    print("Pre-training accuracy on validation set: %.2f\n" % (feed_foward_nn(X_val) == y_val).float().mean())
    print("Starting training...")
    print((X_train[1] - X_train[0]).sum())
    N = X_train.shape[0]
    M = X_val.shape[0]
    for epoch in range(n_epochs):
        feed_foward_nn.train()
        batch_starts = torch.arange(0, N, batch_size)
        tot_acc = 0
        for batch_start in batch_starts:
            batch_end = min(batch_start+batch_size, N)
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            y_pred = feed_foward_nn(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            acc = (y_pred.round() == y_batch).float().mean()
            tot_acc += (batch_end - batch_start) / N * acc
        print("Training accuracy after epoch %d: %.10f" % (epoch, tot_acc))
        feed_foward_nn.eval()
        batch_starts = torch.arange(0, M, batch_size)
        tot_acc = 0
        for batch_start in batch_starts:
            batch_end = min(batch_start+batch_size, M)
            X_batch = X_val[batch_start:batch_end]
            y_batch = y_val[batch_start:batch_end]
            y_pred = feed_foward_nn(X_batch)
            acc = (y_pred.round() == y_batch).float().mean()
            tot_acc += (batch_end - batch_start) / M * acc
        print("Validation accuracy after epoch %d: %.10f\n" % (epoch, tot_acc))
        if tot_acc > best_acc:
            best_acc = tot_acc
            best_weights = copy.deepcopy(feed_foward_nn.state_dict())
        feed_foward_nn.load_state_dict(best_weights) # restore best weights        