import argparse

from src.proteins.protease import PROTEASES
import matplotlib.pyplot as plt
from collections import Counter
from typing import Callable

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
    for target_sequence in sequences:
        peptides = [pep.sequence for pep in protease.cleave(target_sequence)]
        for peptide in peptides:
            if peptide in match_dict:
                match_dict[peptide] += 1
            else:
                match_dict[peptide] = 1
    
    check_len_le_8 = lambda k: len(k) <= 8
    check_len_gt_8 = lambda k: len(k) > 8
    prob_le_8 = compute_prob_cond(match_dict, check_len_le_8)
    prob_gt_8 = compute_prob_cond(match_dict, check_len_gt_8)
    print(f"sequence file: {sequence_file}")
    print(f"probability of non-uniqueness (len <= 8): {prob_le_8}")
    print(f"probability of non-uniqueness (len > 8): {prob_gt_8}")

def compute_prob_cond(match_dict: dict, condition: Callable):
    thresh_dict = {k: v for (k, v) in match_dict.items() if condition(k)}
    tot_peptides = len(thresh_dict.keys())
    n_non_unique = sum([1 for _, v in thresh_dict.items() if v > 1])
    return n_non_unique / tot_peptides

if __name__=="__main__":
    main()