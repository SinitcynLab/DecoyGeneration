import numpy as np
import torch

from torch import Tensor
from typing import Iterable
from collections.abc import Callable

class BaseMetric(object):
    def __init__(self, metric_list: Iterable[Callable], metric_names: Iterable[str], device: torch.device = 'cpu'):
        self.dim = len(metric_list)
        self.metric_list = metric_list
        self.metric_names = metric_names
        self.device = device

    def extract_values(self, predictions: Tensor, targets: Tensor) -> np.ndarray:
        targets_int: Tensor = targets.int()
        out: np.ndarray = np.zeros(self.dim)
        for i, metric in enumerate(self.metric_list):
            out[i] = metric(predictions, targets_int)
        return out
    
    def print_values(self, value_arr: np.ndarray):
        for i, name in enumerate(self.metric_names):
            print(f"{name}: {value_arr[i]:.3f}.")

    def to(self, device: torch.device):
        for metric in self.metric_list:
            if hasattr(metric, 'to'):
                metric.to(device)
        self.device = device