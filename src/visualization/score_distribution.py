import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from typing import Callable, Tuple

def save_scores(scores: Tensor, epoch: int):
    scores_array = scores.numpy()
    np.savetxt(f'data/visualization/val_scores_epoch_{epoch+1}.txt', scores_array, fmt="%.5f")

def graph_score_file(file_name: str, epoch: int, save_name: str, lims: Tuple[float] = None, predicate: Callable[[float], bool] = lambda x: x > -1, ax = None):
    if ax == None:
        plt.figure()
        ax = plt.gca()
    scores_array = np.loadtxt(file_name, dtype=float)
    scores_array = scores_array[predicate(scores_array)]
    sns.histplot(scores_array, bins=25, ax=ax)
    ax.set(title=f"Histogram of scores on epoch {epoch+1}", xlabel='Score', ylabel='Frequency', xlim=lims)
    fig = ax.get_figure()
    fig.savefig(f"{save_name}_{epoch+1}.png")

def graph_all_scores():
    for i in range(0, 20):
        file_name = f'data/visualization/shuffle_scores_rnn/val_scores_epoch_{i+1}.txt'
        graph_score_file(file_name, i, "src/visualization/images/score_distr_epoch")
        graph_score_file(file_name, i, "src/visualization/images/pos_score_distr_epoch", predicate=lambda x: x >= 0.5)
        graph_score_file(file_name, i, "src/visualization/images/neg_score_distr_epoch", predicate=lambda x: x < 0.5)

def graph_score_grid():
    fig, axes = plt.subplots(4, 5, figsize=(20,12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        file_name = f'data/visualization/target_scores_rnn/val_scores_epoch_{i+1}.txt'
        graph_score_file(file_name, i, "src/visualization/images/score_distr_epoch", ax=ax)
    plt.subplots_adjust(hspace=.5) 
    plt.subplots_adjust(left=0.03, right=0.97)
    fig.savefig("src/visualization/images/score_grid.png")

if __name__ == "__main__":
    graph_score_grid()