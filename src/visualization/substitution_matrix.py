import matplotlib.pyplot as plt
import numpy as np
from src.decoy_generators.decoy_generator import DecoyGenerator

if __name__ == "__main__":
    generator = "esm650M.best.c1"
    token_choice_file = f"data/visualization/distr_generators/token_choices_{generator}.txt"
    og_aa_file = f"data/visualization/distr_generators/og_aa_{generator}.txt"
    target_file = f"src/visualization/images/substitution_matrix_{generator}.png"

    labels = DecoyGenerator.canonical_amino_acids
    token_choices = np.loadtxt(token_choice_file, dtype=int)
    og_aas = np.loadtxt(og_aa_file, dtype=int)

    heatmap, xedges, yedges = np.histogram2d(token_choices, og_aas, bins=20)
    
    fig, ax = plt.subplots(figsize=(6,6))
    plt.imshow(
        heatmap.T,
        origin='lower',
        interpolation='nearest'
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Amino acid substituted in")
    ax.set_ylabel("Originally present amino acid")

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax.text(
                i, j, int(heatmap[i,j]),
                ha="center", va="center", color="red", size=8
            )
    plt.title(f"20x20 Matrix showing amino acid substitutions\n for {generator}")

    plt.savefig(target_file)
    plt.show()