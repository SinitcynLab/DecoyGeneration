"""Batch prepare peptides from all FASTA files in ./data.

This script:
  1. Finds every file in the local data/ directory with extension .fasta or .fa
  2. Runs in-silico tryptic digestion (no mass filtering) using prepare_data()
  3. Writes combined unique peptides to peptides.txt

Adjust parameters (min_length, max_length, max_missed_cleavages) below if needed.
Run:  python test.py
"""
from __future__ import annotations
import glob
from pathlib import Path
from decoygen.prepare_data import prepare_data

DATA_DIR = Path('data')
OUT_FILE = Path('peptides.txt')
MIN_LEN = 7
MAX_LEN = 500
MAX_MISSED = None

def collect_fasta_files():
    patterns = [str(DATA_DIR / '*.fasta'), str(DATA_DIR / '*.fa'), str(DATA_DIR / '*.FAA'), str(DATA_DIR / '*.FASTA')]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))

def main():
    fasta_files = collect_fasta_files()
    if not fasta_files:
        print("No FASTA files found in data/. Place .fasta or .fa files there.")
        return
    print(f"Found {len(fasta_files)} FASTA file(s). Digesting...")
    peptides = prepare_data(
        fasta_paths=fasta_files,
        out_path=str(OUT_FILE),
        min_length=MIN_LEN,
        max_length=MAX_LEN,
        max_missed_cleavages=MAX_MISSED,
        mass_min=None,
        mass_max=None,
    )
    print(f"Wrote {len(peptides)} unique peptides to {OUT_FILE}")

if __name__ == '__main__':
    main()