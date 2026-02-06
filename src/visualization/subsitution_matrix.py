import matplotlib.pyplot as plt
import numpy as np
from src.decoy_generators.decoy_generator import DecoyGenerator
from src.visualization.histogram import get_histogram_data

if __name__ == "__main__":
    generator = "esm8M.best.c1"
    token_choice_file = f"data/visualization/distr_generators/token_choices_{generator}.txt"
    og_aa_file = f"data/visualization/distr_generators/og_aa_{generator}.txt"
    target_file = f"src/visualization/images/substitution_matrix_{generator}.png"

    token_choices = np.loadtxt(token_choice_file, dtype=str)
    og_aas = np.loadtxt(og_aa_file, dtype=str)
    labels = DecoyGenerator.canonical_amino_acids

    # prepare histogram data:
    histogram_data = get_histogram_data(og_aas, token_choices, labels)

    # make plot:
    fig, ax = plt.subplots(figsize=(9,9))
    plt.imshow(
        histogram_data,
        origin='lower',
        interpolation='nearest'
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Amino acid originally present at same position")
    ax.set_ylabel(f"Amino acid substituted in")

    for i in range(histogram_data.shape[0]):
        for j in range(histogram_data.shape[1]):
            ax.text(
                i, j, int(histogram_data[i,j]),
                ha="center", va="center", color="red", size=8
            )
    plt.title(f"20x20 Matrix showing amino acid substitutions\n for {generator}")
    plt.savefig(target_file)