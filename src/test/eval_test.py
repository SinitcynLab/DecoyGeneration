import torch

from sklearn.model_selection import train_test_split
from src.peptide_classifiers.mlp_classifier import MLPClassifier, train_mlp
from src.encoders.descriptor_encoder import DescriptionEncoder
from src.encoders.protbert_encoder import ProtBertEncoder
from src.encoders.tokenizer_encoder import TokenizerEncoder
from src.encoders.protalbert_encoder import ProtAlbertEncoder
from src.encoders.esm_encoder import EsmEncoder
from src.io.fasta import read_fasta_file

if __name__ == "__main__":
    # define MLP classifier
    tokenized_len = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(tokenized_len * 1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    encoder = ProtBertEncoder(max_tokenized_length=tokenized_len, device=device)
    classifier = MLPClassifier(network=net, encoder=encoder, device=device)

    # load data:
    target_records = read_fasta_file("data/targets/UP000000625_83333.fasta")
    decoy_records = read_fasta_file("/home/ctrl/DecoyGeneration/data/decoys/UP000000625_83333.shuffle.0.fasta")

    target_sequences = [record.sequence for record in target_records]
    decoy_sequences = [record.sequence for record in decoy_records]
    target_labels = [0 for _ in range(len(target_sequences))]
    decoy_labels = [1 for _ in range(len(decoy_sequences))]

    sequences = target_sequences + decoy_sequences
    labels = target_labels + decoy_labels
    
    test_fraction = 0.3
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=test_fraction)
    
    # train MLP:
    print("Starting training...")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-5)
    n_epochs = 250
    batch_size = 10
    N = 1000
    M = round(N * test_fraction)
    train_mlp(classifier, X_train[1:N], y_train[1:N], X_val[1:M], y_val[1:M], n_epochs, batch_size, optimizer)