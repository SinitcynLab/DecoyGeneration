import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from src.decoy_generators.decoy_generator import DecoyGenerator

if __name__ == "__main__":
    generator = "rel_diff_smart_masking_esm_8M"
    data_file = f"data/visualization/distr_generators/token_choices_{generator}.txt"
    target_file = f"src/visualization/images/token_choices_{generator}.png"
    ordering = DecoyGenerator.canonical_amino_acids

    token_choices = np.loadtxt(data_file, dtype=int)
    freqs = Counter(token_choices)
    freqs = dict(freqs)
    for k in freqs.keys():
        freqs[k] /= len(token_choices) # get relative frequencies
    
    values = list(freqs.values())
    labels = []
    for k in freqs.keys():
        label = DecoyGenerator.canonical_amino_acids[k]
        labels.append(label)
        
    for aa in DecoyGenerator.canonical_amino_acids:
        if aa not in labels:
            labels.append(aa)
            values.append(0.)
    permutation = [labels.index(aa) for aa in ordering]
    bars = plt.bar([labels[i] for i in permutation], [values[i] for i in permutation])

    plt.bar_label(bars, fmt="%.2f", padding=5, fontsize=6)
    plt.xlabel("Amino acid")
    plt.ylabel("Relative frequency")
    plt.title(f"Relative frequency of AAs chosen by {generator}")
    plt.margins(y=0.1)
    plt.savefig(target_file)
    plt.show()