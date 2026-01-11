import torch
import shutil

from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.encoders.spectrum_encoder import SpectrumEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset

def get_mlp_net():
    net = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4000, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    return net

if __name__ == "__main__":
    # define MLP classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.get_num_threads())
    special_amino_acids = ['R', 'K']
    encoder = SpectrumEncoder(special_amino_acids)
    classifier = FeedForwardNNClassifier(network=get_mlp_net(), encoder=encoder, device=device, name="mlp", resetter=get_mlp_net)

    # define MLP classifier
    base = 'UP000000625_83333'
    target_file = f"data/targets/{base}.fasta"
    temp_encoding_dir = f"data/encodings/temp_mlp"

    # target data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    N = len(target_sequences)
    target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path, 1024)

    decoy_files = [f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.esm650M.best.c1.0.fasta']
    decoy_ids = ['shuffle', 'esm650M, count=1']
    
    print("Cross validation of the MLP:")
    for i, decoy_file in enumerate(decoy_files):
        if decoy_file == 'target':
            labels = torch.cat((torch.zeros(N//2), torch.ones(N - N//2)))
            dataset = LMDBDataset([target_lmdb_path], labels)
        else:
            decoy_records = read_fasta_file(decoy_file)
            decoy_sequences = [record.sequence for record in decoy_records]
            M = len(decoy_sequences)
            decoy_lmdb_path = f"{temp_encoding_dir}/{decoy_ids[i]}.lmdb"
            encode_seqs_to_lmdb(decoy_sequences[0:M], encoder, decoy_lmdb_path, 1024)
            labels = torch.cat((torch.zeros(N), torch.ones(M)))
            dataset = LMDBDataset([target_lmdb_path, decoy_lmdb_path], labels)

        # cross-validate MLP:
        n_epochs = 10
        batch_size = 10
        cross_validate_nn(classifier, dataset, n_epochs, batch_size, learning_rate=1e-3, n_folds=5, decoy_id=decoy_ids[i], weight_decay=1e-4)
        if decoy_file != 'target':
            delete_lmdb(decoy_lmdb_path) # clear temporary data
    delete_lmdb(target_lmdb_path)