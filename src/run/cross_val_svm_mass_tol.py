import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.peptide_classifiers.svm_classifier import SVMClassifier, cross_validate_svm
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.encoders.spectrum_encoder import VectorSpectrumEncoder, TupleSpectrumEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset

def mass_tol_kernel(x, y):
    # note that x and y are lists, with each tensor having dimension [1, L, 2]
    sigma = 100.0
    K = len(x)
    L = len(y)
    z = np.zeros((K,L))
    for i in range(K):
        for j in range(L):
            dist = 0
            for k in range(x[i].shape[1]):
                idx = torch.where(abs(x[i][0,k,0] -  y[j][0,:,0]) < sigma)[0] # if mass within tolerance (sigma)
                dist += torch.sum(x[i][0,k,1] * y[j][0,idx,1]) # add products of the intensities to the distance
            z[i, j] = dist
    return z

if __name__ == "__main__":
    # define MLP classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.get_num_threads())
    special_amino_acids = ['R', 'K']
    encoder = TupleSpectrumEncoder(special_amino_acids, normalize_mz = False)
    classifier = SVMClassifier(encoder=encoder, device=device, name="svm", kernel_function=mass_tol_kernel)

    # define MLP classifier
    base = 'UP000000625_83333'
    target_file = f"data/targets/{base}.fasta"
    temp_encoding_dir = f"data/encodings/temp_mlp"

    # target data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    N = len(target_sequences)
    target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path, 512)

    decoy_files = [f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.esm650M.best.c1.0.fasta']
    decoy_ids = ['shuffle', 'esm650M, count=1']
    
    print("Cross validation of the SVM:")
    for i, decoy_file in enumerate(decoy_files):
        if decoy_file == 'target':
            labels = torch.cat((torch.zeros(N//2), torch.ones(N - N//2)))
            dataset = LMDBDataset([target_lmdb_path], labels)
        else:
            decoy_records = read_fasta_file(decoy_file)
            decoy_sequences = [record.sequence for record in decoy_records]
            M = len(decoy_sequences)
            decoy_lmdb_path = f"{temp_encoding_dir}/{decoy_ids[i]}.lmdb"
            encode_seqs_to_lmdb(decoy_sequences[0:M], encoder, decoy_lmdb_path, 512)
            labels = torch.cat((torch.zeros(N), torch.ones(M)))
            dataset = LMDBDataset([target_lmdb_path, decoy_lmdb_path], labels)

        # cross-validate SVM:
        cross_validate_svm(classifier, dataset, n_folds=5)
        if decoy_file != 'target':
            delete_lmdb(decoy_lmdb_path) # clear temporary data
    delete_lmdb(target_lmdb_path)