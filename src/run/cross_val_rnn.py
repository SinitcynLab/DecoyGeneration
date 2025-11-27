import torch

from src.peptide_classifiers.recurrent_nn_classifier import RecurrentNNClassifier
from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.encoders.protbert_encoder import ProtBertEncoder
from src.io.fasta import read_fasta_file
from src.io.utils import split_targets
from src.io.lmdb_writer import encode_seqs_to_lmdb
from src.io.lmdb_dataset import LMDBDataset
import shutil

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
    temp_encoding_dir = f"data/encodings/temp_rnn"

    # target data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    N = len(target_sequences)
    target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path)

    decoy_files = ['target', f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.reverse.fasta']
    decoy_ids = ['target', 'shuffle', 'reverse']
    
    print("Cross validation of the RNN:")
    for i, decoy_file in enumerate(decoy_files):
        if decoy_file == 'target':
            labels = torch.cat((torch.zeros(N//2), torch.ones(N - N//2)))
            dataset = LMDBDataset([target_lmdb_path], labels)
        else:
            decoy_records = read_fasta_file(decoy_file)
            decoy_sequences = [record.sequence for record in decoy_records]
            decoy_lmdb_path = f"{temp_encoding_dir}/{decoy_ids[i]}.lmdb"
            M = len(decoy_sequences)
            encode_seqs_to_lmdb(decoy_sequences[0:M], encoder, decoy_lmdb_path)
            labels = torch.cat((torch.zeros(N), torch.ones(M)))
            dataset = LMDBDataset([target_lmdb_path, decoy_lmdb_path], labels)

        # cross-validate RNN:
        n_epochs = 20
        batch_size = 10
        cross_validate_nn(classifier, dataset, n_epochs, batch_size, learning_rate=1e-3, n_folds=5, decoy_id=decoy_ids[i])
        if decoy_ids[i] != 'target':
            shutil.rmtree(decoy_lmdb_path) # clear temporary data
    shutil.rmtree(target_lmdb_path)