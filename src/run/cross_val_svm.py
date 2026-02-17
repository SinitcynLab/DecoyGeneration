import torch
import time
import datetime

from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity
from typing import Iterable, List
from src.peptide_classifiers.svm_classifier import SVMClassifier, cross_validate_svm
from src.encoders.spectrum_encoder import VectorSpectrumEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset

from src.proteins.protease import Protease


def cos_sim_kernel(x: List[Tensor], y: List[Tensor]):
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    return cosine_similarity(x, y)

def cross_val_svm(target_file: str, decoy_files: Iterable[str], decoy_ids: Iterable[str], protease: Protease):
    # define SVM classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    encoder = VectorSpectrumEncoder(protease)
    classifier = SVMClassifier(encoder=encoder, device=device, name="svm", kernel_function=cos_sim_kernel)

    # define MLP classifier
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    temp_encoding_dir = f"data/encodings/temp_svm_{timestamp}"

    # target data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    N = len(target_sequences)
    target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path, 512)

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