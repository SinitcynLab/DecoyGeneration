import torch
import gc

from collections.abc import Callable
from typing import Iterable, Tuple
from src.peptide_classifiers.nn_classifier import NNClassifier, cross_validate_nn, train_nn
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable
from src.metrics.base_metric import BaseMetric
from src.metrics.default_metric import DefaultMetric
from sklearn.utils import shuffle
from src.io.data_set import Dataset
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

class RecurrentNNClassifier(NNClassifier):
    def __init__(self, rnn : torch.nn.RNN, network : torch.nn.Sequential, encoder : PeptideEncoder, name: str, device : torch.device, resetter: Callable = None):
        NNClassifier.__init__(self, network, encoder, name, device, resetter)
        self.rnn = rnn
        self.rnn.to(self.device)

    def forward(self, data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        N = len(lengths)
        net_inputs = torch.zeros(N, self.rnn.hidden_size).to(self.device)
        for i in range(N):
            rnn_in = data[i, 0:lengths[i], :]
            rnn_out, _ = self.rnn(rnn_in)
            net_inputs[i] = rnn_out[-1, :] # use final output as input for the net
        return torch.flatten(self.network(net_inputs))
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        # get inputs:
        encodings, lengths = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings, lengths)
        # cast outputs to list and return:
        return torch.round(out).tolist()
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> tuple[list[bool], list[float]]:
        # get inputs:
        encodings, lengths = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings, lengths)
        # compute predictions, get accuracy:
        pred : torch.Tensor = torch.round(out)
        outcomes_tensor = torch.IntTensor(list(outcomes)).to(self.device)
        corr : torch.Tensor = torch.eq(pred, outcomes_tensor)
        loss : torch.Tensor = torch.abs(out - outcomes_tensor.float())
        # return whether predictions correct, and loss measure with each prediction:
        return corr.tolist(), loss.tolist()

    def evaluate_on_data(self, dataset: Dataset):
        X, l, y = self.encode_dataset(dataset)
        y_pred = self(X, l)
        return y_pred
    
    def train_on_data(self, dataset: Dataset, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        X, l, y = self.encode_dataset(dataset)
        y_pred = self(X, l)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return y_pred
    
    def set_device(self, device):
        NNClassifier.set_device(self, device)
        self.rnn.to(device)
    
    def reset(self):
        net, rnn = self.resetter()
        self.network = net
        self.rnn = rnn
        self.set_device(self.device)

    def encode_dataset(self, dataset: Dataset):
        seqs, y = dataset.get_contents()
        X, l = self.encoder(seqs)
        return X.to(self.device), l.to(self.device), y.to(self.device)