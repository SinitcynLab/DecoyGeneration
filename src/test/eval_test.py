import torch

from sklearn.model_selection import train_test_split
from src.peptide_classifiers.mlp_classifier import MLPClassifier, train_mlp
from src.encoders.protbert_encoder import ProtBertEncoder
from src.io.fasta import read_fasta_file

if __name__ == "__main__":
    # define MLP classifier
    net = torch.nn.Sequential(
        torch.nn.Linear(1024, 2),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    encoder = ProtBertEncoder()
    classifier = MLPClassifier(net, encoder)

    # load data:
    target_records = read_fasta_file("data/targets/UP000000625_83333.fasta")
    decoy_records = read_fasta_file("/home/ctrl/DecoyGeneration/data/decoys/UP000000625_83333.shuffle.0.fasta")

    target_sequences = [record.sequence for record in target_records]
    decoy_sequences = [record.sequence for record in decoy_records]
    target_labels = [0 for _ in range(len(target_sequences))]
    decoy_labels = [1 for _ in range(len(decoy_sequences))]

    sequences = target_sequences + decoy_sequences
    labels = target_labels + decoy_labels

    # get rid of chains containing 'U' or 'X'
    u_containers = []
    for idx, sequence in enumerate(sequences):
        if 'U' in sequence or 'X' in sequence:
            u_containers.append(idx)
    for idx in sorted(u_containers, reverse=True):
        del sequences[idx]
        del labels[idx]
    
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=0)
    
    # train MLP:
    print("Starting training...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)
    n_epochs = 250
    batch_size = 10
    train_mlp(classifier, X_train, y_train, X_val, y_val, n_epochs, batch_size, optimizer)