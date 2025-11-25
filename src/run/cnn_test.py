import torch

from sklearn.model_selection import train_test_split
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.peptide_classifiers.nn_classifier import train_nn
from src.encoders.image_encoder import ImageEncoder
from src.io.fasta import read_fasta_file

if __name__ == "__main__":
    # define CNN classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=(3,1)),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(4, 16, kernel_size=(3,1)),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Flatten(),
        torch.nn.Linear(16*64*254, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid()
    )
    # dim calculation:
    # O = ((W - K + 2 * P) / S) + 1
    # O: output size
    # W: Input size
    # K: kernel size
    # P: padding (0 for our case)
    # S: stride
    encoder = ImageEncoder(image_height=256, device=device)
    classifier = FeedForwardNNClassifier(network=net, encoder=encoder, device=device, name="cnn")

    base = 'UP000002311_559292'

    # load data:
    target_records = read_fasta_file(f"data/targets/{base}.fasta")
    decoy_records = read_fasta_file(f"/home/ctrl/DecoyGeneration/data/decoys/{base}.esm8M.best.[0.05].0.fasta")

    target_sequences = [record.sequence for record in target_records]
    decoy_sequences = [record.sequence for record in decoy_records]

    target_labels = [0 for _ in range(len(target_sequences))]
    decoy_labels = [1 for _ in range(len(decoy_sequences))]

    sequences = target_sequences + decoy_sequences
    labels = target_labels + decoy_labels
    
    test_fraction = 0.3
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=test_fraction)
    
    # train CNN:
    N = 500 # 500 each
    M = round(N * test_fraction)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    n_epochs = 20
    batch_size = 10
    best_acc = train_nn(classifier, X_train[0:N], y_train[0:N], X_val[0:M], y_val[0:M], n_epochs, batch_size, optimizer)
    print(f"Best accuracy of CNN on {base}: {best_acc}")