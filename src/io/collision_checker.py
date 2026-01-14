import argparse
import sys

from src.io.fasta import read_fasta_file
from src.io.peptide_processor import PeptideProcessor

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--targets", help="The file holding the target sequences.")
    parser.add_argument("-d", "--decoys", help="The file holding the decoy sequences.")
    parser.add_argument("-m", "--min_length", help="The minimum length of peptides to take into account.", default=-1)
    parser.add_argument("-M", "--max_length", help="The maximum length of peptides to take into account.", default=sys.maxsize)

    return parser.parse_args()

def main():
    args = collect_args()
    target_file = args.targets
    decoy_file = args.decoys
    min_len = args.min_length
    max_len = args.max_length
    peptide_processor = PeptideProcessor(['L', 'R'])
    
    target_records = read_fasta_file(target_file)
    decoy_records = read_fasta_file(decoy_file)
    
    target_sequences = [rec.sequence for rec in target_records]
    decoy_sequences = [rec.sequence for rec in decoy_records]
    tot_collisions, tot_peptides = 0, 0
    for i, decoy_sequence in enumerate(decoy_sequences):
        for target_sequence in target_sequences:
            collisions, n_peptides = count_collisions(decoy_sequence, target_sequence, peptide_processor, min_len, max_len)
            tot_collisions += collisions
            tot_peptides += n_peptides
        if i % 1000 == 0:
            print(f"{i+1}/{len(decoy_sequences)}")
    print(f"target file: {target_file}")
    print(f"decoy file: {decoy_file}")
    print(f"Number of collisions: {tot_collisions}")
    print(f"Total number of peptides with specified length: {tot_peptides}")

def count_collisions(decoy_sequence: str, target_sequence: str, peptide_processor: PeptideProcessor, min_len: int, max_len: int):
    count: int = 0
    decoy_peptides = peptide_processor.get_all_peptides(decoy_sequence)
    decoy_peptides = set(decoy_sequence[peptide[0]:peptide[-1]] for peptide in decoy_peptides if min_len <= len(peptide) <= max_len)
    target_peptides = peptide_processor.get_all_peptides(target_sequence)
    target_peptides = set(target_sequence[peptide[0]:peptide[-1]] for peptide in target_peptides if min_len <= len(peptide) <= max_len)

    for decoy_peptide in decoy_peptides:
        if decoy_peptide in target_peptides:
            count += 1

    return count, len(decoy_peptides)

if __name__=="__main__":
    main()