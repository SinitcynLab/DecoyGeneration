import matplotlib.pyplot as plt
import numpy as np
from src.decoy_generators.decoy_generator import DecoyGenerator
from src.visualization.histogram import get_histogram_data

if __name__ == "__main__":
    generator = "esm650M.best.c1"
    type = "most_probable_aa" # token_choices or most_probable_aa
    token_choice_file = f"data/visualization/distr_generators/{type}_{generator}.txt"
    offset_file = f"data/visualization/distr_generators/aa_offset_{generator}.txt"

    labels = DecoyGenerator.canonical_amino_acids
    token_choices = np.loadtxt(token_choice_file, dtype=str, delimiter=",")
    og_aas = np.loadtxt(offset_file, dtype=str, delimiter=",")
    og_aas_min = og_aas[:,0]
    og_aas_plus = og_aas[:,1]
    names = ["minus", "plus"]
    name_to_og_data: dict = {names[0]: og_aas_min, names[1]: og_aas_plus}

    for name in names:
        # create and fill dict with data:
        target_file = f"src/visualization/images/offset_{name}_matrix_{generator}.png"
        og_aas = name_to_og_data[name]

        histogram_data = get_histogram_data(og_aas, token_choices, labels)

        # make plot:
        fig, ax = plt.subplots(figsize=(9,9))
        plt.imshow(
            histogram_data.T,
            origin='lower',
            interpolation='nearest'
        )
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"Originally present amino acid at position i {name} 1")
        ax.set_ylabel("Amino acid judged to be most likely at position")

        for i in range(histogram_data.shape[0]):
            for j in range(histogram_data.shape[1]):
                ax.text(
                    i, j, int(histogram_data[i,j]),
                    ha="center", va="center", color="red", size=8
                )
        plt.title(f"20x20 Matrix showing amino acid substitutions\n for {generator}, offset {name} 1")
        plt.savefig(target_file)