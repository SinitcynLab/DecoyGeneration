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