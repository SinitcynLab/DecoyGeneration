import torch

from src.metrics.base_metric import BaseMetric
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

class DefaultMetric(BaseMetric):
    def __init__(self):
        auroc = AUROC(task='binary')
        accuracy = BinaryAccuracy()
        precision = BinaryPrecision()
        recall = BinaryRecall()
        BaseMetric.__init__(self, [auroc, accuracy, precision, recall], ['AUC', 'Accuracy', 'Precision', 'Recall'])