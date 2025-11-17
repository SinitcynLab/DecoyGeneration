import torch
import copy
import numpy as np

from src.peptide_classifiers.nn_classifier import NNClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable

class RecurrentNNClassifier(NNClassifier):
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