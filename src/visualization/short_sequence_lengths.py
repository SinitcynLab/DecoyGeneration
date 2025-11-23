import matplotlib.pyplot as plt

from src.io.fasta import read_fasta_file

if __name__ == "__main__":
    recs = read_fasta_file("data/targets/UP000005640_9606.fasta") # human protein data
    lens = [len(rec.sequence) for rec in recs]

    bin_edges = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    plt.hist(lens, edgecolor='black', bins=bin_edges)
    plt.title("Histogram of sequence lengths in human protein data.")
    plt.xlabel("Sequence length")
    plt.ylabel("Frequency")

    plt.show()