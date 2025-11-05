import torch

from src.peptide_classifiers.nn_classifier import NNClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable

class LSTMClassifier(NNClassifier, torch.nn.Module):
    def __init__(self, lstm : torch.nn.LSTM, network : torch.nn.Sequential, encoder : PeptideEncoder, device : torch.device):
        NNClassifier.__init__(self, network, encoder, device)
        self.lstm = lstm
        self.lstm.to(self.device)

    def forward(self, tensor_list : Iterable[torch.Tensor], h0 : torch.Tensor = None, c0 : torch.Tensor = None) -> torch.Tensor:
        N = len(tensor_list)
        outputs = torch.zeros(N).to(self.device)
        for i, t in enumerate(tensor_list):
            if h0 is None or c0 is None:
                rnn_out, _ = self.lstm(t) # h0, c0 default to zero tensors of appropriate dimension
            else:
                rnn_out, _ = self.lstm(t, (h0, c0))
            output = self.network(rnn_out[:, -1, :]) # take output at final step
            outputs[i] = output
        return outputs