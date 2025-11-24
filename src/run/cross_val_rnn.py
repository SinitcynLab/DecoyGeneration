import torch

from src.peptide_classifiers.recurrent_nn_classifier import RecurrentNNClassifier
from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.encoders.protbert_encoder import ProtBertEncoder
from src.io.fasta import read_fasta_file
from src.io.utils import split_targets

def get_rnn_net():
    out_size = 2048
    rnn = torch.nn.RNN(1024, out_size, bidirectional=False)
    net = torch.nn.Sequential( # each character (amino acid) is encoded using 1024 numbers
        torch.nn.Linear(out_size, 1),
        torch.nn.Sigmoid()
    )
    return net, rnn

if __name__ == "__main__":
    # define MLP classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.get_num_threads())
    encoder = ProtBertEncoder(device=device, constant_length=False, flatten=False)
    net, rnn = get_rnn_net()
    classifier = RecurrentNNClassifier(rnn=rnn, network=net, encoder=encoder, device=device, name="rnn", resetter=get_rnn_net)

    base = 'UP000002311_559292'
    target_file = f"data/targets/{base}.fasta"

    # load data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]

    decoy_files = ['target', f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.reverse.fasta',
                   f'data/decoys/{base}.diann_C.fasta', f'data/decoys/{base}.diann_random_pos.fasta',f'data/decoys/{base}.diann_N.fasta']
    decoy_ids = ['target', 'shuffle', 'reverse', 'diann_C', 'diann_random_pos', 'diann_N']
    
    print("Cross validation of the RNN:")
    for i, decoy_file in enumerate(decoy_files):
        if decoy_file == 'target':
            target_sequences, decoy_sequences = split_targets(target_sequences)
        else:
            decoy_records = read_fasta_file(decoy_file)
            decoy_sequences = [record.sequence for record in decoy_records]

        target_labels = [0 for _ in range(len(target_sequences))]
        decoy_labels = [1 for _ in range(len(decoy_sequences))]

        # cross-validate RNN:
        n_epochs = 20
        batch_size = 2
        sequences = target_sequences + decoy_sequences
        labels = target_labels + decoy_labels
        cross_validate_nn(classifier, sequences, labels, n_epochs, batch_size, 1e-3, n_folds=5, decoy_id=decoy_ids[i])