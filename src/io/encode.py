import argparse
import torch
import lmdb
import pickle
import shutil
import os
import numpy as np

from typing import Iterable

from src.io.fasta import read_fasta_file
from src.encoders.protbert_encoder import ProtBertEncoder

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="File with proteins to encode. Can be only one file, but you can choose multiple encodings..")
    parser.add_argument("-e", "--encodings", nargs="+", help="Encodings to use. If you use multiple, must be separated by spaces (e.g. a b c).")
    parser.add_argument("-l", "--length", help="Length of encoding to use.")
    parser.add_argument("-o", "--output_files", nargs="+", help="Files to which you write the encoded input files. This command encodes input to output files respectively. Filenames must be separated by spaces (e.g. a b c).")

    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = collect_args()
    i_file = args.input_file
    o_files = args.output_files
    encodings = args.encodings

    if len(encodings) != len(o_files):
        raise ValueError("Encode only accepts input and output lists of equal length, and encodes input to output files respectively.")
    fasta_records = read_fasta_file(i_file)
    sequences = [record.sequence for record in fasta_records]

    for encoding, o_file in zip(encodings, o_files):
        if os.path.exists(o_file):
            shutil.rmtree(o_file)
        if encoding == 'recurrent':
            encoder = ProtBertEncoder(device=device, constant_length=False, flatten=False)
            recurrent_seqs_to_lmdb(encoder, sequences, device, o_file)
        else:
            raise ValueError("Specify a valid encoder.")

def recurrent_seqs_to_lmdb(encoder: ProtBertEncoder, sequences: Iterable[str], device: torch.device, o_file_name: str):
    BATCH_SIZE = 32
    batch_starts = np.arange(0, len(sequences), BATCH_SIZE)
    for batch_start in batch_starts:
        batch_end = min(len(sequences), batch_start + BATCH_SIZE)
        batch_encodings = encoder(sequences[batch_start:batch_end])
        append_tensors_to_lmbdb(batch_encodings, range(batch_start, batch_end), o_file_name)
        print(f"{batch_end}/{len(sequences)}")

def append_tensors_to_lmbdb(tensors: Iterable[torch.Tensor], indices: Iterable[int], out_file: str):
    env = lmdb.open(out_file, map_size=1024**4)
    with env.begin(write=True) as txn:
        for (i, t) in zip(indices, tensors):
            key = f"{i}".encode()
            txn.put(key, pickle.dumps(t))

if __name__=="__main__":
    main()