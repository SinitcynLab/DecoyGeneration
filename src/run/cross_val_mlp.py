import torch
import shutil
import datetime
import time

from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.encoders.spectrum_encoder import VectorSpectrumEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset
from sklearn.utils import shuffle

def get_mlp_net():
    net = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(1024, 128),
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
    print(torch.get_num_threads())
    encoder = ProtBertClsEncoder(device)
    classifier = FeedForwardNNClassifier(network=get_mlp_net(), encoder=encoder, device=device, name="mlp", resetter=get_mlp_net)

    # define MLP classifier
    base = 'UP000002311_559292'
    target_file = f"data/targets/{base}.fasta"
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    temp_encoding_dir = f"data/encodings/temp_mlp_{timestamp}"

    # target data:
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    N = len(target_sequences)
    #target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    #encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path)

    decoy_files = [f'data/decoys/{base}.reverse.fasta', f'data/decoys/{base}.shuffle.0.fasta',
                   f'data/decoys/{base}.diann_C.fasta', f'data/decoys/{base}.esm8M.best.c1.0.fasta',
                   f'data/decoys/{base}.esm650M.best.c1.0.fasta']
    decoy_ids = ['reverse', 'shuffle', 'diann_C', 'esm 8M, count=1, 32bit', 'esm 650M, count=1, 32bit']
    
    print("Cross validation of the MLP:")
    for i, decoy_file in enumerate(decoy_files):
        if decoy_file == 'target':
            labels = torch.cat((torch.zeros(N//2), torch.ones(N - N//2)))
            #dataset = LMDBDataset([target_lmdb_path], labels)
        else:
            decoy_records = read_fasta_file(decoy_file)
            decoy_sequences = [record.sequence for record in decoy_records]
            M = len(decoy_sequences)
            decoy_lmdb_path = f"{temp_encoding_dir}/{decoy_ids[i]}.lmdb"
            sequences = target_sequences[0:N] + decoy_sequences[0:M]            
            labels = [0 for _ in range(N)] + [1 for _ in range(M)]
            sequences, labels = shuffle(sequences, labels)
            encode_seqs_to_lmdb(sequences, encoder, decoy_lmdb_path)
            dataset = LMDBDataset([decoy_lmdb_path], torch.FloatTensor(labels))

        # cross-validate MLP:
        n_epochs = 10
        batch_size = 10
        cross_validate_nn(classifier, dataset, n_epochs, batch_size, learning_rate=1e-3, n_folds=5, decoy_id=decoy_ids[i])
        if decoy_file != 'target':
            delete_lmdb(decoy_lmdb_path) # clear temporary data
    #delete_lmdb(target_lmdb_path)