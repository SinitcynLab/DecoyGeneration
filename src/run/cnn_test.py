import torch

from sklearn.model_selection import train_test_split
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.peptide_classifiers.nn_classifier import train_nn
from src.encoders.image_encoder import ImageEncoder
from src.io.fasta import read_fasta_file
from src.io.utils import split_targets

def get_cnn_net():
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
    return net

if __name__ == "__main__":
    # define CNN classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ImageEncoder(image_height=256, device=device)
    classifier = FeedForwardNNClassifier(network=get_cnn_net(), encoder=encoder, device=device, name="cnn", resetter=get_cnn_net)

    base = 'UP000002311_559292'
    test_fraction = 0.3

    # load data:
    target_records = read_fasta_file(f"data/targets/{base}.fasta")
    decoy_records = read_fasta_file(f"/home/ctrl/DecoyGeneration/data/decoys/{base}.esm8M.best.[0.05].0.fasta")

    # load data:
    target_sequences = [record.sequence for record in target_records]

    decoy_files = ['target', f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.reverse.fasta',
                   f'data/decoys/{base}.diann_C.fasta', f'data/decoys/{base}.esm8M.best.[0.05].0.fasta', f'data/decoys/{base}.esm8M.best.[0.3].0.fasta',
                   f'data/decoys/{base}.esm650M.best.[0.05].0.(1233).fasta', f'data/decoys/{base}.esm650M.best.[0.3].0.(2190).fasta']
    decoy_ids = ['target', 'shuffle', 'reverse', 'diann_C', 'diann_random_pos', 'diann_N']
    
    for i, decoy_file in enumerate(decoy_files):
        classifier.reset()
        optimizer = torch.optim.Adam(params=classifier.parameters(), lr = 1e-3)
        if decoy_file == 'target':
            target_sequences, decoy_sequences = split_targets(target_sequences)
        else:
            decoy_records = read_fasta_file(decoy_file)
            decoy_sequences = [record.sequence for record in decoy_records]

        target_labels = [0 for _ in range(len(target_sequences))]
        decoy_labels = [1 for _ in range(len(decoy_sequences))]
        
        # cross-validate MLP:
        n_epochs = 20
        batch_size = 10
        N = 500 #len(target_labels) + len(decoy_labels)
        sequences = target_sequences[0:N] + decoy_sequences[0:N]
        labels = target_labels[0:N] + decoy_labels[0:N]

        X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=test_fraction)

        print(f"results for {decoy_ids[i]}")
        train_nn(classifier, X_train, y_train, X_val, y_val, n_epochs, batch_size, optimizer)