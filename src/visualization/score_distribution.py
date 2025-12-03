import seaborn as sns
import numpy as np

from torch import Tensor

def save_scores(scores: Tensor, epoch: int):
    scores_array = scores.numpy()
    np.savetxt(f'data/visualization/val_scores_epoch_{epoch+1}.txt', scores_array, fmt="%.5f")

def graph_score_file(file_name: str, epoch: int):
    scores_array = np.loadtxt(file_name, dtype=float)
    sns.set_style('whitegrid')
    plot = sns.kdeplot(scores_array, bw=0.5, x='Score', y='Density estimate')
    plot.set_title(f"Scores on epoch {epoch+1}")
    fig = plot.get_figure()
    fig.savefig(f"src/visualization/images/{file_name}.png")

def graph_all_scores(N: int):
    for i in range(N):
        file_name = f'data/visualization/val_scores_epoch_{i+1}.txt'
        graph_score_file(file_name, i)

if __name__ == "__main__":
    graph_all_scores(20)