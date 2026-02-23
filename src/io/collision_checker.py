import argparse
import sys

from typing import List, Tuple
from statistics import mean
from src.proteins.protease import PROTEASES

from src.io.fasta import read_fasta_file

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--targets", help="The file holding the target sequences.")
    parser.add_argument("-d", "--decoys", help="The file holding the decoy sequences.")
    parser.add_argument("-p", "--protease", default="trypsin", help="The protease used for splitting up the proteins.")

    return parser.parse_args()

def main():
    args = collect_args()
    target_file = args.targets
    decoy_file = args.decoys
    protease = PROTEASES[args.protease]

    target_records = read_fasta_file(target_file)
    decoy_records = read_fasta_file(decoy_file)
    
    target_sequences = [rec.sequence for rec in target_records]
    decoy_sequences = [rec.sequence for rec in decoy_records]

    target_peptides = set()
    decoy_peptides = list()
    for target_sequence in target_sequences:
        peptides = [pep.sequence for pep in protease.cleave(target_sequence)]
        target_peptides = target_peptides | set(peptides) # note '|' is the union operator
    for decoy_sequence in decoy_sequences:
        peptides = [pep.sequence for pep in protease.cleave(decoy_sequence)]
        decoy_peptides += peptides

    result_list: List[dict] = []
    for decoy_peptide in decoy_peptides:
        if decoy_peptide in target_peptides:
            result_list.append({"collided": 1, "length": len(decoy_peptide)})
        else:
            result_list.append({"collided": 0, "length": len(decoy_peptide)})

    print(f"Target file: {target_file}")
    print(f"Decoy file: {decoy_file}")
    print()
    print("Stats for all peptides:")
    print_stats(result_list)
    print()
    print("Stats for peptides length <= 8:")
    print_stats(result_list, (-1, 8))
    print()
    print("Stats for peptides length > 8:")
    print_stats(result_list, (9, sys.maxsize))
    print()

def print_stats(result_list: List[dict], len_range: Tuple[int, int] = (-1, sys.maxsize)):
    filtered_list = [result for result in result_list if len_range[0] <= result["length"] <= len_range[1]]

    collided_list = [result['collided'] for result in filtered_list]
    len_list = [result['length'] for result in filtered_list]
    
    print(f"Probability of peptide colliding (estimate): {mean(collided_list)}")
    print(f"Mean peptide length: {mean(len_list)}")

if __name__=="__main__":
    main()