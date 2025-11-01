import torch
import copy
import numpy as np

from src.peptide_classifiers.nn_classifier import NNClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable

class RecurrentNNClassifier(NNClassifier, torch.nn.Module):
    def __init__(self, rnn : torch.nn.RNN, network : torch.nn.Sequential, encoder : PeptideEncoder, device : torch.device):
        NNClassifier.__init__(self, network, encoder, device)
        self.rnn = rnn
        self.rnn.to(self.device)

    def forward(self, tensor_list : list[torch.Tensor]) -> torch.Tensor:
        outputs = torch.zeros(len(tensor_list)).to(self.device)
        for i, t in enumerate(tensor_list):
            rnn_out, _ = self.rnn(t)
            output = self.network(rnn_out[:, -1, :]) # take output at final step
            outputs[i] = output
        return outputs
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        # get inputs as list:
        tensor_list : list[torch.Tensor] = self.encoder(sequences)
        # iterate over inputs and store outputs:
        out : torch.Tensor = torch.zeros(len(sequences))
        for i, t in enumerate(tensor_list):
            out[i] = self.forward(t)
        # cast outputs to list and return:
        res : list[bool] = torch.round(tensor_list).tolist()
        return res
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> tuple[list[bool], list[float]]:
        # get inputs as list:
        tensor_list : list[torch.Tensor] = self.encoder(sequences)
        # iterate over inputs and store outputs:
        out : torch.Tensor = torch.zeros(len(sequences))
        for i, t in enumerate(tensor_list):
            out[i] = self.forward(t)
        # compute predictions, get accuracy:
        pred : torch.Tensor = torch.round(out)
        outcomes_tensor = torch.IntTensor(list(outcomes)).to(self.device)
        corr : torch.Tensor = torch.eq(pred, outcomes_tensor)
        loss : torch.Tensor = torch.abs(out - outcomes_tensor.float())
        # return whether predictions correct, and loss measure with each prediction:
        return corr.tolist(), loss.tolist()
    
def train_recurrent_nn(recurrent_nn : RecurrentNNClassifier, X_train : Iterable[str], y_train : Iterable[bool], X_val : Iterable[str], 
              y_val : Iterable[str], n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    X_train = recurrent_nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).to(recurrent_nn.device)
    X_val = recurrent_nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).to(recurrent_nn.device)

    loss_fn = torch.nn.BCELoss()

    best_acc = - np.inf
    best_weights = None

    print("Starting training...")
    N = len(X_train)
    M = len(X_val)
    for epoch in range(n_epochs):
        recurrent_nn.train()
        batch_starts = torch.arange(0, N, batch_size)
        tot_acc = 0
        for batch_start in batch_starts:
            batch_end = min(batch_start + batch_size, N)
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            y_pred = recurrent_nn(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            acc = (y_pred.round() == y_batch).float().mean()
            tot_acc += (batch_end - batch_start) / N * acc
        print("Training accuracy after epoch %d: %.3f" % (epoch, tot_acc))
        recurrent_nn.eval()
        batch_starts = torch.arange(0, M, batch_size)
        tot_acc = 0
        for batch_start in batch_starts:
            batch_end = min(batch_start + batch_size, M)
            X_batch = X_val[batch_start:batch_end]
            y_batch = y_val[batch_start:batch_end]
            y_pred = recurrent_nn(X_batch)
            acc = (y_pred.round() == y_batch).float().mean()
            tot_acc += (batch_end - batch_start) / M * acc
        print("Validation accuracy after epoch %d: %.3f\n" % (epoch, tot_acc))
        if tot_acc > best_acc:
            best_acc = tot_acc
            best_weights = copy.deepcopy(recurrent_nn.state_dict())
    recurrent_nn.load_state_dict(best_weights) # restore best weights
    print("Best validation accuracy recorded: %.3f" % best_acc) 