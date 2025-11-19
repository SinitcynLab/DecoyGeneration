import torch
import copy
import numpy as np

from collections.abc import Callable
from typing import Iterable
from src.peptide_classifiers.nn_classifier import NNClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable

class RecurrentNNClassifier(NNClassifier):
    def __init__(self, rnn : torch.nn.RNN, network : torch.nn.Sequential, encoder : PeptideEncoder, name: str, device : torch.device, resetter: Callable = None):
        NNClassifier.__init__(self, network, encoder, name, device, resetter)
        self.rnn = rnn
        self.rnn.to(self.device)

    def forward(self, tensor_list : Iterable[torch.Tensor]) -> torch.Tensor:
        outputs = torch.zeros(len(tensor_list)).to(self.device)
        for i, t in enumerate(tensor_list):
            rnn_out, _ = self.rnn(t)
            output = self.network(rnn_out[:, -1, :]) # take output at final step
            outputs[i] = output
        return outputs
    
    def set_device(self, device):
        NNClassifier.set_device(self, device)
        self.rnn.to(device)
    
    def reset(self):
        net, rnn = self.resetter()
        self.network = net
        self.rnn = rnn
        self.set_device(self.device)