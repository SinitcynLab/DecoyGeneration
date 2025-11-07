import torch

from sklearn.model_selection import train_test_split
from src.peptide_classifiers.lstm_classifier import LSTMClassifier
from src.peptide_classifiers.nn_classifier import train_nn
from src.encoders.protbert_encoder import ProtBertEncoder
from src.encoders.esm_encoder import ESMEncoder
from src.io.fasta import read_fasta_file

if __name__ == "__main__":
    # define RNN classifier
    hidden_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm = torch.nn.LSTM(320, hidden_size, 1)
    net = torch.nn.Sequential( # each character (amino acid) is encoded using 1024 numbers
        torch.nn.Linear(hidden_size, 1),
        torch.nn.Sigmoid()
    )
    encoder = ESMEncoder(device=device, constant_length=False, flatten=False)
    classifier = LSTMClassifier(lstm=lstm, network=net, encoder=encoder, device=device)

    base = 'UP000000625_83333'

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
    
    # train LSTM:
    N = 3000 # 100 each
    M = round(N * test_fraction)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    n_epochs = 20
    batch_size = 10
    best_acc = train_nn(classifier, X_train[0:N], y_train[0:N], X_val[0:M], y_val[0:M], n_epochs, batch_size, optimizer)
    print(f"Best accuracy of LSTM on {base}: {best_acc}")