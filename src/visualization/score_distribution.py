import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from typing import Callable

def save_scores(scores: Tensor, epoch: int):
    scores_array = scores.numpy()
    np.savetxt(f'data/visualization/val_scores_epoch_{epoch+1}.txt', scores_array, fmt="%.5f")

def graph_score_file(file_name: str, epoch: int, save_name: str, predicate: Callable[[float], bool] = lambda x: x > -1):
    scores_array = np.loadtxt(file_name, dtype=float)
    scores_array = scores_array[predicate(scores_array)]
    plt.figure()
    sns.set_style('whitegrid')
    plot = sns.kdeplot(scores_array)
    plot.set(title=f"Scores on epoch {epoch+1}", xlabel='Score', ylabel='Density estimate', xlim=(0,1))
    fig = plot.get_figure()
    fig.savefig(f"{save_name}_{epoch+1}.png")

def graph_all_scores(N: int):
    for i in range(15, N):
        file_name = f'data/visualization/val_scores_epoch_{i+1}.txt'
        graph_score_file(file_name, i, "src/visualization/images/score_distr_epoch")
        graph_score_file(file_name, i, "src/visualization/images/pos_score_distr_epoch", lambda x: x >= 0.5)
        graph_score_file(file_name, i, "src/visualization/images/neg_score_distr_epoch", lambda x: x < 0.5)

if __name__ == "__main__":
    graph_all_scores(20)