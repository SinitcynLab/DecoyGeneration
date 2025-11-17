import torch
import copy
import numpy as np

from typing import Iterable, Union
from src.peptide_classifiers.peptide_classifier import PeptideClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

class NNClassifier(PeptideClassifier, torch.nn.Module):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder, name: str, device : torch.device):
        PeptideClassifier.__init__(self, encoder, name, device)
        torch.nn.Module.__init__(self)
        self.network = network
        network.to(device)
        self.auroc = AUROC(task='binary').to(self.device)
        self.accuracy = BinaryAccuracy().to(self.device)
        self.precision = BinaryPrecision().to(self.device)
        self.recall = BinaryRecall().to(self.device)

    def set_device(self, device : torch.device):
        PeptideClassifier.set_device(device)
        self.network.to(device)

    def evaluate_on_batch(self, X_batch, y_batch) -> tuple[torch.Tensor, float]:
        y_pred = self(X_batch)
        return y_pred

    def train_on_batch(self, X_batch, y_batch, loss_fn, optimizer) -> float:
        y_pred = self.evaluate_on_batch(X_batch, y_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return y_pred
    
    # perhaps following two methods should be moved to separate 'metrics' class, but this seems overkill for now
    def extract_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        int_targets = targets.int()
        auc = self.auroc(predictions, int_targets)
        acc = self.accuracy(predictions, int_targets)
        precision = self.precision(predictions, int_targets)
        recall = self.recall(predictions, int_targets)
        return torch.tensor([auc, acc, precision, recall]).cpu().numpy()
    
    def print_metrics(self, metrics:np.ndarray, label_metrics:str):
        if label_metrics not in ['training', 'validation']:
            raise ValueError("Please provide an appropriate label for the metrics ('training' or 'validation')")
        print(f"Best {label_metrics} ROC of {self}: {metrics[0]:.3f} (Note: can be mean over folds).")
        print(f"Corresponding {label_metrics} accuracy of {self}: {metrics[1]:.3f}.")
        print(f"Corresponding {label_metrics} precision of {self}: {metrics[2]:.3f}.")
        print(f"Corresponding {label_metrics} recall of {self}: {metrics[3]:.3f}.")
    
    def classify(self, sequences : Iterable[str]) -> list[bool]:
        # get inputs:
        encodings : Union[torch.Tensor, list[torch.Tensor]] = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings)
        # cast outputs to list and return:
        return torch.round(out).tolist()
    
    def score(self, sequences : Iterable[str], outcomes : Iterable[bool]) -> tuple[list[bool], list[float]]:
        # get inputs:
        encodings : Union[torch.Tensor, list[torch.Tensor]] = self.encoder(sequences) # could be single tensor or list of tensor, depending on implementation of encoder
        # get outputs through forward
        out = self.forward(encodings)
        # compute predictions, get accuracy:
        pred : torch.Tensor = torch.round(out)
        outcomes_tensor = torch.IntTensor(list(outcomes)).to(self.device)
        corr : torch.Tensor = torch.eq(pred, outcomes_tensor)
        loss : torch.Tensor = torch.abs(out - outcomes_tensor.float())
        # return whether predictions correct, and loss measure with each prediction:
        return corr.tolist(), loss.tolist()

def set_up_nn_training(nn : NNClassifier, X_train, y_train, X_val, y_val):
    X_train = nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).unsqueeze(1).to(nn.device)
    X_val = nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).unsqueeze(1).to(nn.device)

    loss_fn = torch.nn.BCELoss()

    best_acc = - np.inf
    best_weights = None

    return X_train, y_train, X_val, y_val, loss_fn, best_acc, best_weights

def cross_validate_nn(nn: NNClassifier, sequences: Iterable[str], labels: Iterable[str], 
                   n_epochs: int, batch_size: int, 
                   optimizer: torch.optim.Optimizer, n_folds: int = 5) -> float:
    sequences, labels = shuffle(sequences, labels)
    data: torch.Tensor = nn.encoder(sequences)
    labels: torch.Tensor = torch.FloatTensor(list(labels))

    loss_fn = torch.nn.BCELoss()
    mean_best_val_metrics: np.ndarray = np.zeros(4) # [auc, acc, prec, rec]
    mean_corr_train_metrics: np.ndarray = np.zeros(4)

    kfold = StratifiedKFold(n_splits=n_folds)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data.cpu(), labels)):
        best_val_metrics: np.ndarray = np.ones(4) * (- np.inf) 
        corr_train_metrics: np.ndarray = np.zeros(4)
        train_data: torch.Tensor = data[train_ids].to(nn.device)
        val_data: torch.Tensor = data[val_ids].to(nn.device)
        train_labels: torch.Tensor = labels[train_ids].to(nn.device)
        val_labels: torch.Tensor = labels[val_ids].to(nn.device)

        for _ in range(n_epochs):
            train_metrics, val_metrics = train_val_iteration(nn, train_data, val_data, train_labels,
                                                              val_labels, loss_fn, optimizer, batch_size)
            # use ROC as criterion:
            if val_metrics[0] > best_val_metrics[0]: 
                best_val_metrics = val_metrics
                corr_train_metrics = train_metrics

        print(f"Validation metrics over fold #{fold + 1}:")
        nn.print_metrics(best_val_metrics, 'validation')

        mean_best_val_metrics += best_val_metrics / n_folds
        mean_corr_train_metrics += corr_train_metrics / n_folds

    print(f"")
    print(f"### Average over all {n_folds} folds: ###")
    print(f"--- Validation ---")
    nn.print_metrics(mean_best_val_metrics, 'validation')
    print(f"--- Training ---")
    nn.print_metrics(mean_corr_train_metrics, 'training')

    return mean_best_val_metrics[0] # return mean recorded 'best' ROC


def train_val_iteration(nn: NNClassifier, train_data: torch.Tensor, val_data: torch.Tensor, 
                        train_labels: torch.Tensor, val_labels: torch.Tensor, loss, 
                        optimizer:torch.optim.Optimizer, batch_size: int):
    # train:
    N: int = len(train_data)
    nn.train()
    batch_starts: np.ndarray = np.arange(0, N, batch_size)
    predictions: torch.Tensor = torch.zeros(N).to(nn.device)
    for batch_start in batch_starts:
        batch_end: int = min(batch_start + batch_size, N)
        batch_data: torch.Tensor = train_data[batch_start:batch_end]
        batch_labels: torch.Tensor = train_labels[batch_start:batch_end]
        y_pred = nn.train_on_batch(batch_data, batch_labels, loss, optimizer)
        predictions[batch_start:batch_end] = y_pred
    avg_train_metrics = nn.extract_metrics(predictions, train_labels)

    # validate:
    M: int = len(val_data)
    avg_val_metrics: np.ndarray = np.zeros(4)
    nn.eval()
    batch_starts: np.ndarray = np.arange(0, M, batch_size)
    predictions: torch.Tensor = torch.zeros(M).to(nn.device)
    for batch_start in batch_starts:
        batch_end: int = min(batch_start + batch_size, M)
        batch_data: torch.Tensor = val_data[batch_start:batch_end]
        batch_labels: torch.Tensor = val_labels[batch_start:batch_end]
        y_pred = nn.evaluate_on_batch(batch_data, batch_labels)
        predictions[batch_start:batch_end] = y_pred
    avg_val_metrics = nn.extract_metrics(predictions, val_labels)

    return avg_train_metrics, avg_val_metrics

def train_nn(nn : NNClassifier, X_train : Iterable[str], y_train : Iterable[bool], X_val : Iterable[str], 
              y_val : Iterable[str], n_epochs : int, batch_size : int, optimizer : torch.optim.Optimizer):
    X_train = nn.encoder(X_train)
    y_train = torch.FloatTensor(list(y_train)).to(nn.device)
    X_val = nn.encoder(X_val)
    y_val = torch.FloatTensor(list(y_val)).to(nn.device)
    
    loss_fn = torch.nn.BCELoss()

    best_val_acc = - np.inf
    best_weights = None

    N = len(X_train)
    M = len(X_val)
    for epoch in range(n_epochs):
        _, val_acc = train_val_iteration(nn, X_train, y_train, X_val, y_val, loss_fn, optimizer, batch_size)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(nn.state_dict())

    nn.load_state_dict(best_weights) # restore best weights
    return best_val_acc # return best validation accuracy