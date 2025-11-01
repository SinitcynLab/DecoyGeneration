import torch
import numpy as np

from src.peptide_classifiers.peptide_classifier import PeptideClassifier
from src.encoders.peptide_encoder import PeptideEncoder

class NNClassifier(PeptideClassifier, torch.nn.Module):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder, device : torch.device):
        PeptideClassifier.__init__(self, encoder, device)
        torch.nn.Module.__init__(self)
        self.network = network
        network.to(device)

    def set_device(self, device : torch.device):
        PeptideClassifier.set_device(device)
        self.network.to(device)

def set_up_nn_training(nn : NNClassifier, X_train, y_train, X_val, y_val):
    X_train = nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).unsqueeze(1).to(nn.device)
    X_val = nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).unsqueeze(1).to(nn.device)

    loss_fn = torch.nn.BCELoss()

    best_acc = - np.inf
    best_weights = None

    return X_train, y_train, X_val, y_val, loss_fn, best_acc, best_weights