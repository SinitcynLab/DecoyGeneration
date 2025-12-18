import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    generator = "esm650M.best.c1"
    data_file = f"data/visualization/distr_generators/prob_distr_{generator}.txt"
    target_file = f"src/visualization/images/prob_distr_{generator}.png"

    with open (data_file) as file:
        content = file.read().strip()

    blocks = content.split("\n\n")

    mean_first = 0
    mean_second = 0
    for block in blocks:
        arr = np.loadtxt(block.splitlines())
        mean_first += arr[0]
        mean_second += arr[1]
    
    mean_first /= len(blocks)
    mean_second /= len(blocks)

    labels = ['Mean mass original acid', 'Mean mass substituted acid']
    values = [mean_first, mean_second]
    bars = plt.bar(labels, values)
    plt.bar_label(bars, padding=-20)
    plt.xlabel("Type of amino acid")
    plt.ylabel("Probability mass")
    plt.title(f"Mean probability masses of original, using {generator}")
    plt.savefig(target_file)
    plt.show()