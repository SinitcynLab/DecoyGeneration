import torch
import numpy as np

from collections.abc import Callable
from typing import List, Tuple
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from torch import Tensor

from src.peptide_classifiers.peptide_classifier import PeptideClassifier
from src.encoders.peptide_encoder import PeptideEncoder
from src.io.lmdb_dataset import LMDBDataset
from src.metrics.base_metric import BaseMetric
from src.metrics.default_metric import DefaultMetric

class SVMClassifier(PeptideClassifier):
    def __init__(self, network : torch.nn.Sequential, encoder : PeptideEncoder, name: str, device : torch.device, kernel : Callable = None):
        PeptideClassifier.__init__(self, encoder, name, device)
        torch.nn.Module.__init__(self)
        self.network = network
        self.network.to(device)
        self.kernel = kernel
        self.svm_instance = svm.SVC(kernel=kernel)

    def set_device(self, device : torch.device):
        PeptideClassifier.set_device(self, device)
        self.network.to(device)

    def evaluate_on_data(self, dataset: Tuple[List[Tensor], Tensor]):
        val_data = torch.cat(dataset[0], dim=0)
        prediction_tensor = torch.tensor(self.svm_instance.predict(val_data))
        return prediction_tensor

    def train_on_data(self, dataset: Tuple[List[Tensor], Tensor]):
        train_data, train_labels = torch.cat(dataset[0], dim=0), dataset[1].numpy()
        self.svm_instance.fit(train_data, train_labels)
    
    def reset(self):
        self.svm_instance = svm.SVC(kernel=self.kernel)

def cross_validate_svm(svm: SVMClassifier, main_dataset: LMDBDataset, n_folds: int = 5, metric: BaseMetric = DefaultMetric()):
    N = main_dataset.size()

    kfold = StratifiedKFold(n_splits=n_folds)
    val_metrics = np.zeros((n_folds, metric.dim))
    train_metrics = np.zeros((n_folds, metric.dim))
    for fold, (train_ids, val_ids) in enumerate(kfold.split(torch.zeros(N), main_dataset.get_labels())):
        train_ids = shuffle(train_ids)
        val_ids = shuffle(val_ids)
        
        # reset and train:
        svm.reset()
        train_dataset = main_dataset.get_pairs(train_ids)
        svm.train_on_data(train_dataset)

        # evaluate:
        val_dataset = main_dataset.get_pairs(val_ids)
        validation_predictions = svm.evaluate_on_data(val_dataset)
        val_metrics_fold = metric.extract_values(validation_predictions, val_dataset[1])
        train_predictions = svm.evaluate_on_data(train_dataset)
        train_metrics_fold = metric.extract_values(train_predictions, train_dataset[1])

        print(f"Validation AUC over #{fold + 1}, with other corresponding metrics:")
        metric.print_metric(val_metrics_fold)
        print(f"Validation set info on fold #{fold + 1}:")
        print(f"Size: {len(val_ids)}.")
        print(f"N.o. targets: {main_dataset.get_num_targets(val_ids)}.")
        print(f"N.o. decoys: {main_dataset.get_num_decoys(val_ids)}.")

        val_metrics[fold,:] = val_metrics_fold
        train_metrics[fold,:] = train_metrics_fold
        print(f"{fold}/{n_folds}")

    print(f"### Average best validation AUC and corresponding validation metrics over all {n_folds} folds: ###")
    metric.print_metric_series(val_metrics)
    print(f"### Average corresponding training metrics over all {n_folds} folds: ###")
    metric.print_metric_series(train_metrics)

    print(f"")