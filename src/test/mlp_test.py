import torch
import numpy as np

from sklearn.model_selection import train_test_split
from src.peptide_classifiers.nn_classifier import train_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier, train_feed_forward_nn
from src.encoders.protbert_pca_encoder import ProtBertPcaEncoder
from src.io.fasta import read_fasta_file

if __name__ == "__main__":
    # define RNN classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenized_len = 64
    pca_dim = 100
    net = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(pca_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    pca_encoder = ProtBertPcaEncoder(max_tokenized_length=tokenized_len, device=device, pca_dim=pca_dim)
    classifier = FeedForwardNNClassifier(network=net, encoder=pca_encoder, device=device)

    base = 'UP000002311_559292'

    # load data:
    target_records = read_fasta_file(f"data/targets/{base}.fasta")
    decoy_records = read_fasta_file(f"/home/ctrl/DecoyGeneration/data/decoys/{base}.reverse.fasta")

    target_sequences = [record.sequence for record in target_records]
    decoy_sequences = [record.sequence for record in decoy_records]

    target_labels = [0 for _ in range(len(target_sequences))]
    decoy_labels = [1 for _ in range(len(decoy_sequences))]

    sequences = target_sequences + decoy_sequences
    labels = target_labels + decoy_labels
    
    test_fraction = 0.3
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=test_fraction)
    
    # train MLP:
    N = 100 # 100 each
    M = round(N * test_fraction)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    n_epochs = 20
    batch_size = 10
    best_acc = train_nn(classifier, X_train[0:N], y_train[0:N], X_val[0:M], y_val[0:M], n_epochs, batch_size, optimizer)
    print(f"Best accuracy of MLP on {base}: {best_acc}")