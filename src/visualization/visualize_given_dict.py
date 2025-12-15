import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from src.decoy_generators.decoy_generator import DecoyGenerator

from src.io.fasta import read_fasta_file

if __name__ == "__main__":
    target_file = f"src/visualization/images/true_token_freqs.png"

    freq_dict = {'M': 0.020864142557172387, 'S': 0.08988429349173019, 'Y': 0.03385337941440006, 
            'T': 0.05914553936908157, 'D': 0.05835266995116823, 'N': 0.06157044656693329, 
            'P': 0.043781098592239755, 'Q': 0.03949708452825461, 'K': 0.07337497514836089, 
            'R': 0.04444664591384583, 'A': 0.05488875997810332, 'L': 0.09512220197778196, 
            'V': 0.05556043510113596, 'H': 0.02172305605712745, 'E': 0.06520048804534029, 
            'G': 0.0496671922958557, 'F': 0.04436460146141255, 'I': 0.06560458250608014, 
            'W': 0.010402964221810071, 'C': 0.012695442822165755}
    ordering = DecoyGenerator.canonical_amino_acids
    values = list(freq_dict.values())
    labels = list(freq_dict.keys())
    
    for label in ordering:
        if label not in labels:
            labels.append(label)
            values.append(0.)
    permutation = [labels.index(aa) for aa in ordering]
    bars = plt.bar([labels[i] for i in permutation], [values[i] for i in permutation])

    plt.bar_label(bars, fmt="%.2f", padding=5, fontsize=6)
    plt.xlabel("Amino acid")
    plt.ylabel("Relative frequency")
    plt.title(f"Relative frequencies of various AAs in the original target data")
    plt.margins(y=0.1)
    plt.savefig(target_file)
    plt.show()
    print(freq_dict.keys())