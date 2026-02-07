import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from src.decoy_generators.decoy_generator import DecoyGenerator

if __name__ == "__main__":
    generator = "esm8M.best.c1"
    data_file = f"data/visualization/distr_generators/token_choices_{generator}.txt"
    target_file = f"src/visualization/images/token_choices_{generator}.png"
    ordering = DecoyGenerator.canonical_amino_acids

    token_choices = np.loadtxt(data_file, dtype=str)
    freqs = Counter(token_choices)
    freqs = dict(freqs)
    for k in freqs.keys():
        freqs[k] /= len(token_choices) # get relative frequencies
    
    # cart the dictionary:
    bars = plt.bar(freqs.keys(), freqs.values())

    plt.bar_label(bars, fmt="%.2f", padding=5, fontsize=6)
    plt.xlabel("Amino acid")
    plt.ylabel("Relative frequency")
    plt.title(f"Relative frequency of AAs chosen by {generator}")
    plt.margins(y=0.1)
    plt.savefig(target_file)
    plt.show()