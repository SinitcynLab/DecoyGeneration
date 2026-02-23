import argparse

from src.proteins.protease import PROTEASES
import matplotlib.pyplot as plt
from collections import Counter

from src.io.fasta import read_fasta_file

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sequence_file", help="The file holding the sequences.")
    parser.add_argument("-p", "--protease", default="trypsin", help="The protease used for splitting up the proteins.")
    parser.add_argument("-g", "--graph", action="store_true", help="Whether to graph the result.")

    return parser.parse_args()

def main():
    args = collect_args()
    sequence_file = args.sequence_file
    protease = PROTEASES[args.protease]

    records = read_fasta_file(sequence_file)
    
    sequences = [rec.sequence for rec in records]

    match_dict = dict()
    tot_peptides = 0
    for target_sequence in sequences:
        peptides = [pep.sequence for pep in protease.cleave(target_sequence)]
        for peptide in peptides:
            tot_peptides += 1
            if peptide in match_dict:
                match_dict[peptide] += 1
            else:
                match_dict[peptide] = 1
    
    n_non_unique = sum([1 for _, v in match_dict.items() if v > 1])
    prob_non_unique = n_non_unique / tot_peptides
    print(f"sequence file: {sequence_file}")
    print(f"probability of non-uniqueness: {prob_non_unique}")
    print()
    if args.graph:
        counts = Counter(match_dict.values())
        x = [1, 2, 3, 4]
        y = [counts[i] for i in x]
        plt.bar(x, y, edgecolor='black')
        plt.xlabel("Repeat count")
        plt.ylabel("Number of peptides with this repeat count")
        plt.title("Histogram of 'repeat count'")
        plt.savefig("src/visualization/images/repeat_count_histogram.png")

if __name__=="__main__":
    main()