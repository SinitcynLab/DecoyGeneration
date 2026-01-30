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
from src.encoders.transformer_encoder import pad_tensor_list
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

    def evaluate_on_data(self, tensor_list: Iterable[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            X, l = pad_tensor_list(tensor_list)
            # note that the tensor list is converted to tensors only because it makes removal from VRAM simpler
            # if you keep the tensors in the list instead, then having the list reference persists also keeps tensors in VRAM
            X, l = X.to(self.device), l.to(self.device)
            y_pred = self(X, l)
            del X, l
            torch.cuda.empty_cache()
            return y_pred
    
    def train_on_data(self, tensor_list: Iterable[torch.Tensor], y: torch.Tensor, 
                    loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        X, l = pad_tensor_list(tensor_list)
        # note that the tensor list is converted to tensors only because it makes removal from VRAM simpler
        # if you keep the tensors in the list instead, then having the list reference persists also keeps tensors in VRAM
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
        NNClassifier.set_device(self, device)
        self.rnn.to(device)
    
    def reset(self):
        net, rnn = self.resetter()
        self.network = net
        self.rnn = rnn
        self.set_device(self.device)