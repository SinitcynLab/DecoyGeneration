import torch
import numpy as np

from sklearn.model_selection import train_test_split
from src.peptide_classifiers.recurrent_nn_classifier import RecurrentNNClassifier
from src.peptide_classifiers.nn_classifier import train_nn
from src.encoders.protbert_encoder import ProtBertEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset
import datetime
import time

if __name__ == "__main__":
    # define RNN classifier
    out_size = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn = torch.nn.RNN(1024, out_size, bidirectional=False)
    net = torch.nn.Sequential( # each character (amino acid) is encoded using 1024 numbers
        torch.nn.Linear(out_size, 1),
        torch.nn.Sigmoid()
    )
    encoder = ProtBertEncoder(device=device, constant_length=False, flatten=False)
    classifier = RecurrentNNClassifier(rnn=rnn, network=net, encoder=encoder, device=device, name="rnn")

    base = 'UP000002311_559292'

    target_file = f"data/targets/{base}.fasta"
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    temp_encoding_dir = f"data/encodings/temp_rnn_{timestamp}"
    decoy_file = f'data/decoys/{base}.shuffle.0.fasta'
    decoy_id = 'esm650M[count=1]'

    # target data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    N = len(target_sequences)
    target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path)

    # overall dataset:
    labels = torch.cat((torch.zeros(N//2), torch.ones(N - N//2)))
    dataset = LMDBDataset([target_lmdb_path], labels)

    # train-test split:
    test_fraction = 0.3
    train_ids, val_ids, _, _ = train_test_split(np.arange(N), labels, test_size=test_fraction)

    print("Evaluation of the RNN:")
    n_epochs = 20
    batch_size = 10
    train_nn(classifier, dataset, train_ids, val_ids, n_epochs, batch_size, learning_rate=1e-3)
    delete_lmdb(target_lmdb_path)
