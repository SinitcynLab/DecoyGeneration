import torch
import datetime
import time

from typing import Iterable, Callable

from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.encoders.esm_cls_encoder import EsmClsEncoder
from src.encoders.peptide_encoder import PeptideEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset

def get_mlp(dim: int):
    net = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    return net

def cross_val_mlp_esm(target_file: str, decoy_files: Iterable[str], decoy_ids: Iterable[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EsmClsEncoder(device)
    get_mlp_esm = lambda : get_mlp(1280)
    __cross_val_mlp(target_file, decoy_files, decoy_ids, encoder, get_mlp_esm, device)

def cross_val_mlp_protbert(target_file: str, decoy_files: Iterable[str], decoy_ids: Iterable[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ProtBertClsEncoder(device)
    get_mlp_protbert = lambda : get_mlp(1024)
    __cross_val_mlp(target_file, decoy_files, decoy_ids, encoder, get_mlp_protbert, device)
     

def __cross_val_mlp(target_file: str, decoy_files: Iterable[str], decoy_ids: Iterable[str], encoder: PeptideEncoder, resetter: Callable, device: torch.device):
        # define MLP classifier
        print(f"Using {device}...")
        classifier = FeedForwardNNClassifier(network=resetter(), encoder=encoder, device=device, name="mlp", resetter=resetter)

        # define MLP classifier
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        temp_encoding_dir = f"data/encodings/temp_mlp_{timestamp}"

        # target data:
        target_records = read_fasta_file(target_file)
        target_sequences = [record.sequence for record in target_records]
        N = len(target_sequences)
        target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
        encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path)

        print("Cross validation of the MLP:")
        for i, decoy_file in enumerate(decoy_files):
            if decoy_file == 'target':
                labels = torch.cat((torch.zeros(N//2), torch.ones(N - N//2)))
                dataset = LMDBDataset([target_lmdb_path], labels)
            else:
                decoy_records = read_fasta_file(decoy_file)
                decoy_sequences = [record.sequence for record in decoy_records]
                M = len(decoy_sequences)
                decoy_lmdb_path = f"{temp_encoding_dir}/{decoy_ids[i]}.lmdb"
                encode_seqs_to_lmdb(decoy_sequences[0:M], encoder, decoy_lmdb_path)
                labels = torch.cat((torch.zeros(N), torch.ones(M)))
                dataset = LMDBDataset([target_lmdb_path, decoy_lmdb_path], labels)

            # cross-validate MLP:
            n_epochs = 10
            batch_size = 10
            cross_validate_nn(classifier, dataset, n_epochs, batch_size, learning_rate=1e-3, n_folds=5, decoy_id=decoy_ids[i])
            if decoy_file != 'target':
                delete_lmdb(decoy_lmdb_path) # clear temporary data
        delete_lmdb(target_lmdb_path)