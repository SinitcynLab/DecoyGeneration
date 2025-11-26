import argparse
import torch
import lmdb
import pickle
import numpy as np
import os
import shutil

from typing import Iterable

from src.io.fasta import read_fasta_file
from src.encoders.protbert_encoder import ProtBertEncoder
from src.encoders.peptide_encoder import PeptideEncoder

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_files", nargs="+", help="Files with proteins to encode. If you specify multiple, then they must be separated by spaces (e.g. a b c).")
    parser.add_argument("-e", "--encoding_type", help="Encoder to use")
    parser.add_argument("-l", "--length", help="Length of encoding to use.")
    parser.add_argument("-o", "--output_files", nargs="+", help="Files to which you write the encoded input files. This command encodes input to output files respectively. Filenames must be separated by spaces (e.g. a b c).")

    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = collect_args()
    i_file_names = args.input_files
    o_file_names = args.output_files

    if len(i_file_names) != len(o_file_names):
        raise ValueError("Encode only accepts input and output lists of equal length, and encodes input to output files respectively.")
    file_pairs = zip(i_file_names, o_file_names)
    if args.encoding_type == 'recurrent':
        encoder = ProtBertEncoder(device=device, constant_length=False, flatten=False)
    else:
        raise ValueError("Specify a valid encoder.")

    for i_file, o_file in zip(file_pairs):
        fasta_records = read_fasta_file(i_file)
        sequences = [record.sequence for record in fasta_records]
        encode_seqs_to_lmdb(sequences, encoder, o_file, device)

def encode_seqs_to_lmdb(sequences: Iterable[str], encoder: PeptideEncoder, o_file_name: str):
    if os.path.exists(o_file_name) and os.path.isdir(o_file_name):
        shutil.rmtree(o_file_name)
    os.makedirs(o_file_name)
    BATCH_SIZE = 32
    batch_starts = np.arange(0, len(sequences), BATCH_SIZE)
    for batch_start in batch_starts:
        batch_end = min(len(sequences), batch_start + BATCH_SIZE)
        batch_encodings = encoder(sequences[batch_start:batch_end])
        append_tensors_to_lmbdb(batch_encodings, range(batch_start, batch_end), o_file_name)

def append_tensors_to_lmbdb(tensors: Iterable[torch.Tensor], indices: Iterable[int], out_file: str):
    env = lmdb.open(out_file, map_size=1024**4)
    pairs = zip(indices, tensors) # note that this will iterate over list of tensors/first dim of tensor containing batch
    with env.begin(write=True) as txn:
        for (i, t) in pairs:
            key = f"{i}".encode()
            txn.put(key, pickle.dumps(t))

if __name__=="__main__":
    main()