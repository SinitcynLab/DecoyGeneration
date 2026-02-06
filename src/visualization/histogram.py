import numpy as np

from typing import List

def get_histogram_data(x_array: np.ndarray, y_array: np.ndarray, labels: List[str]):
    # create and fill dict with data:
    data: dict = {}
    for i in range(len(x_array)):
        key = (x_array[i], y_array[i])
        if key in data:
            data[key] += 1
        else:
            data[key] = 1
    
    # extract histogram from dict:
    K = len(labels)
    histogram = np.zeros((K, K), dtype=int)
    for (x, y), v in data.items():
        if '' in [x, y]:
            continue # In this case, the datapoint is considered invalid (usually means i+1 or i-1 was out of sequence range)
        i, j = labels.index(x), labels.index(y)
        histogram[i, j] += v
    return histogram