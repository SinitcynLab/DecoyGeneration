import torch
import numpy as np

from sklearn.model_selection import train_test_split
from src.peptide_classifiers.recurrent_nn_classifier import RecurrentNNClassifier
from src.peptide_classifiers.nn_classifier import train_nn
from src.encoders.protbert_encoder import ProtBertEncoder
from src.io.fasta import read_fasta_file

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
    target_path = f"data/targets/{base}.fasta"
    decoy_path = f"data/decoys/{base}.shuffle.[0.75].0.fasta"

    # load data:
    target_records = read_fasta_file(target_path)
    decoy_records = read_fasta_file(decoy_path)
    
    #K = 2190
    target_sequences = [record.sequence for record in target_records]
    decoy_sequences = [record.sequence for record in decoy_records]
    target_sequences = target_sequences
    decoy_sequences = decoy_sequences

    target_labels = [0 for _ in range(len(target_sequences))]
    decoy_labels = [1 for _ in range(len(decoy_sequences))]

    sequences = target_sequences + decoy_sequences
    labels = target_labels + decoy_labels
    
    test_fraction = 0.3
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=test_fraction)
    
    # train RNN:
    N = 40 # 100 each
    M = round(N * test_fraction)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    n_epochs = 20
    batch_size = 10
    best_acc = train_nn(classifier, X_train[0:N], y_train[0:N], X_val[0:M], y_val[0:M], n_epochs, batch_size, optimizer)
    print(f"Best accuracy of RNN on {decoy_path}: {best_acc}")