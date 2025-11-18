import torch

from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.io.fasta import read_fasta_file
from src.io.utils import split_targets

if __name__ == "__main__":
    # define MLP classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(1024, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    encoder = ProtBertClsEncoder(device=device)
    classifier = FeedForwardNNClassifier(network=net, encoder=encoder, device=device, name="mlp")

    base = 'UP000002311_559292'
    target_file = f"data/targets/{base}.fasta"

    # load data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]

    decoy_files = ['target', f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.reverse.fasta',
                   f'data/decoys{base}.diann_C.fasta', f'data/decoys{base}.diann_random_pos.fasta',f'data/decoys{base}.diann_N.fasta']
    decoy_ids = ['target', 'shuffle', 'reverse', 'diann_C', 'diann_random_pos', 'diann_N']
    
    for i, decoy_file in enumerate(decoy_files):
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
        sequences = target_sequences + decoy_sequences
        labels = target_labels + decoy_labels
        cross_validate_nn(classifier, sequences, labels, n_epochs, batch_size, 1e-3, n_folds=5, decoy_id=decoy_ids[i])