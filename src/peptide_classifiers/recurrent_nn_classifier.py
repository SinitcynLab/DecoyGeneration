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

    def forward(self, tensor_list: Iterable[torch.Tensor]) -> torch.Tensor:
        outputs = torch.zeros(len(tensor_list)).to(self.device)
        for i, t in enumerate(tensor_list):
            rnn_out, _ = self.rnn(t)
            outputs[i] = self.network(rnn_out[:,-1,:]) # use last output as network input
        return outputs
    
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
        tensor_list, _ = self.encode_dataset(dataset)
        y_pred = self(tensor_list)
        del tensor_list
        return y_pred
    
    def train_on_data(self, dataset: Dataset, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        tensor_list, y = self.encode_dataset(dataset)
        y_pred = self(tensor_list)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        del tensor_list
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
        tensor_list = self.encoder(seqs)
        for i in range(len(tensor_list)):
            tensor_list[i] = tensor_list[i].to(self.device)
        return tensor_list, y.to(self.device)