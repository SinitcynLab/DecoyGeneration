import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    generator = "esm650M.best.c1"
    data_file = f"data/visualization/distr_generators/prob_distr_{generator}.txt"
    target_file = f"src/visualization/images/prob_distr_{generator}.png"

    with open (data_file) as file:
        content = file.read().strip()

    blocks = content.split("\n\n")

    mean_og = 0
    mean_chosen = 0
    mean_max = 0
    for block in blocks:
        arr = np.loadtxt(block.splitlines())
        mean_og += arr[0]
        mean_chosen += arr[1]
        mean_max += arr[2]
    
    mean_og /= len(blocks)
    mean_chosen /= len(blocks)
    mean_max /= len(blocks)

    labels = ['Mean probability \noriginal acid', 'Mean probability\n substituted acid', 'Mean probability amino\n acid with max prob.']
    values = [mean_og, mean_chosen, mean_max]
    bars = plt.bar(labels, values)
    plt.bar_label(bars, padding=-20)
    plt.ylabel("Probability mass")
    plt.title(f"Mean probability masses of various AAs, using {generator}")
    plt.savefig(target_file)
    plt.show()