import torch

from typing import Iterable, Callable

from src.peptide_classifiers.plm_free_classifier import PlmFreeClassifier
from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.encoders.custom_tokenizer_encoder import CustomTokenizer
from src.proteins.aminoacids import AMINOACIDS, EXTRA_AMINOACIDS
from src.proteins.protease import Protease
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset
import datetime
import time

def get_plm_free(dim: int, pad_id: int):
    embedding = torch.nn.Embedding(dim, 128, pad_id)
    rnn = torch.nn.GRU(
        input_size=64,
        hidden_size=256,
        num_layers=3,
        batch_first=True,
        bidirectional=True
    )
    net = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(128, 1)
    )
    return net, rnn, embedding

def get_peptide_number(sequences: Iterable[str], protease: Protease):
    tot_peptide_num: int = 0
    for sequence in sequences:
        tot_peptide_num += len(protease.cleave(sequence))
    return tot_peptide_num

def cross_val_plm_free(target_file: str, decoy_files: Iterable[str], decoy_ids: Iterable[str], protease: Protease):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")

    # define encoder:
    amino_acids = list(AMINOACIDS + EXTRA_AMINOACIDS)
    encoder = CustomTokenizer(amino_acids=amino_acids, protease=protease, peptide_level=True)

    # define classifier:
    resetter = lambda: get_plm_free(encoder.vocab_size, encoder.pad_id)
    net, rnn, embedding = resetter()
    classifier = PlmFreeClassifier(rnn=rnn, network=net, embedding=embedding, encoder=encoder, device=device, name="rnn", resetter=resetter)

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    temp_encoding_dir = f"data/encodings/temp_rnn_{timestamp}"

    # target data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    N = get_peptide_number(target_sequences, protease)
    encode_seqs_to_lmdb(target_sequences, encoder, target_lmdb_path)

    print(f"Cross validation of the PLM-free classifier (peptide_level={encoder.peptide_level}):")
    for i, decoy_file in enumerate(decoy_files):
        if decoy_file == 'target':
            labels = torch.cat((torch.zeros(N//2), torch.ones(N - N//2)))
            dataset = LMDBDataset([target_lmdb_path], labels)
        else:
            decoy_records = read_fasta_file(decoy_file)
            decoy_sequences = [record.sequence for record in decoy_records]
            decoy_lmdb_path = f"{temp_encoding_dir}/{decoy_ids[i]}.lmdb"
            M = get_peptide_number(decoy_sequences, protease)
            encode_seqs_to_lmdb(decoy_sequences, encoder, decoy_lmdb_path)
            labels = torch.cat((torch.zeros(N), torch.ones(M)))
            dataset = LMDBDataset([target_lmdb_path, decoy_lmdb_path], labels)

        # cross-validate RNN:
        n_epochs = 5
        batch_size = 1024
        cross_validate_nn(classifier, dataset, n_epochs, batch_size, learning_rate=1e-3, n_folds=5, decoy_id=decoy_ids[i])
        if decoy_ids[i] != 'target':
            delete_lmdb(decoy_lmdb_path) # clear temporary data
    delete_lmdb(target_lmdb_path)