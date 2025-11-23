import torch

from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier, cross_validate_ff_nn
from src.encoders.esm_cls_encoder import ESMCLSEncoder
from src.io.fasta import read_fasta_file

def get_mlp():
    net = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(320, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    return net

if __name__ == "__main__":
    # define MLP classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    encoder = ESMCLSEncoder(device=device)
    classifier = FeedForwardNNClassifier(network=get_mlp(), encoder=encoder, device=device, name="mlp", resetter=get_mlp)

    base = 'UP000002311_559292'
    target_file = f"data/targets/{base}.fasta"
    decoy_file = f"/home/ctrl/DecoyGeneration/data/decoys/{base}.esm8M.best.[0.05].0.fasta"

    # load data:
    target_records = read_fasta_file(target_file)
    decoy_records = read_fasta_file(decoy_file)

    target_sequences = [record.sequence for record in target_records]
    decoy_sequences = [record.sequence for record in decoy_records]

    target_labels = [0 for _ in range(len(target_sequences))]
    decoy_labels = [1 for _ in range(len(decoy_sequences))]

    # cross-validate MLP:
    N = 100
    n_epochs = 20
    batch_size = 10
    sequences = target_sequences[0:N//2] + decoy_sequences[0:N//2]
    labels = target_labels[0:N//2] + decoy_labels[0:N//2]
    cross_validate_ff_nn(classifier, sequences, labels, n_epochs, batch_size, 1e-3, n_folds=5, decoy_id='esm8')