import torch
import numpy as np

from src.peptide_classifiers.svm_classifier import SVMClassifierUMAP, cross_validate_svm
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.encoders.spectrum_encoder import VectorSpectrumEncoder, SmoothVectorSpectrumEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset

def linear_kernel(x, y):
    return x @ y.T
    

if __name__ == "__main__":
    # define MLP classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.get_num_threads())
    special_amino_acids = ['R', 'K']
    encoder = VectorSpectrumEncoder(special_amino_acids)
    classifier = SVMClassifierUMAP(encoder=encoder, device=device, name="svm", kernel_function=linear_kernel)

    # define MLP classifier
    base = 'human_and_crap'
    target_file = f"data/targets/{base}.fasta"
    temp_encoding_dir = f"data/encodings/temp_mlp"

    # target data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    N = 15000#len(target_sequences)
    target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path, 512)

    decoy_files = [f'data/decoys/{base}.shuffle.[0].0.fasta', f'data/decoys/{base}.esm650M.best.c1.0.fasta']
    decoy_ids = ['shuffle', 'esm650M, count=1']
    
    print("Cross validation of the SVM:")
    for i, decoy_file in enumerate(decoy_files):
        if decoy_file == 'target':
            labels = torch.cat((torch.zeros(N//2), torch.ones(N - N//2)))
            dataset = LMDBDataset([target_lmdb_path], labels)
        else:
            decoy_records = read_fasta_file(decoy_file)
            decoy_sequences = [record.sequence for record in decoy_records]
            M = 15000#len(decoy_sequences)
            decoy_lmdb_path = f"{temp_encoding_dir}/{decoy_ids[i]}.lmdb"
            encode_seqs_to_lmdb(decoy_sequences[0:M], encoder, decoy_lmdb_path, 512)
            labels = torch.cat((torch.zeros(N), torch.ones(M)))
            dataset = LMDBDataset([target_lmdb_path, decoy_lmdb_path], labels)

        # cross-validate SVM:
        cross_validate_svm(classifier, dataset, n_folds=5)
        if decoy_file != 'target':
            delete_lmdb(decoy_lmdb_path) # clear temporary data
    delete_lmdb(target_lmdb_path)