import torch

from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.image_encoder import ImageEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset

def get_cnn_net():
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(4, 16, kernel_size=4),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Flatten(),
        torch.nn.Linear(16*62*254, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid()
    )
    return net

if __name__ == "__main__":
    # define CNN classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ImageEncoder(image_height=256, device=device)
    classifier = FeedForwardNNClassifier(network=get_cnn_net(), encoder=encoder, device=device, name="cnn", resetter=get_cnn_net)

    # define data
    base = 'UP000002311_559292'
    target_file = f"data/targets/{base}.fasta"
    temp_encoding_dir = f"data/encodings/temp_cnn"

    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    N = len(target_sequences)
    encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path)

    decoy_files = ['target', f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.reverse.fasta',
                   f'data/decoys/{base}.diann_C.fasta', f'data/decoys/{base}.esm8M.best.[0.05].0.fasta']
    decoy_ids = ['target', 'shuffle', 'reverse', 'diann', 'esm8M[0.05]']
    
    print("Cross validation of the CNN:")
    for i, decoy_file in enumerate(decoy_files):
        if decoy_file == 'target':
            labels = torch.cat((torch.zeros(N//2), torch.ones(N - N//2)))
            dataset = LMDBDataset([target_lmdb_path], labels)
        else:
            decoy_records = read_fasta_file(decoy_file)
            decoy_sequences = [record.sequence for record in decoy_records]
            M = len(decoy_sequences)
            decoy_lmdb_path = f"{temp_encoding_dir}/{decoy_ids[i]}.lmdb"
            encode_seqs_to_lmdb(decoy_sequences[0:M], encoder, decoy_lmdb_path)
            labels = torch.cat((torch.zeros(N), torch.ones(M)))
            dataset = LMDBDataset([target_lmdb_path, decoy_lmdb_path], labels)

        # cross-validate CNN:
        n_epochs = 20
        batch_size = 10
        cross_validate_nn(classifier, dataset, n_epochs, batch_size, learning_rate=1e-3, n_folds=5, decoy_id=decoy_ids[i])
        if decoy_ids[i] != 'target':
            delete_lmdb(decoy_lmdb_path) # clear temporary data
    delete_lmdb(target_lmdb_path)