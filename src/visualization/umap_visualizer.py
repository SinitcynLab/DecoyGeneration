import torch
import numpy as np
import umap
from umap import plot
import matplotlib.pyplot as plt

from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.encoders.spectrum_encoder import VectorSpectrumEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset

def generate_umap_image(targets: LMDBDataset, decoys: LMDBDataset, path: str):
    target_data, _ = targets.get_pairs(range(targets.size()))
    decoy_data, _ = decoys.get_pairs(range(decoys.size()))
    max_dim: int = max([torch.numel(t) for t in target_data] + [torch.numel(t) for t in decoy_data])
    tot_size = targets.size() + decoys.size()
    
    plot_data: np.ndarray = np.zeros((tot_size, max_dim))
    plot_labels: np.ndarray = np.zeros(tot_size)
    for i in range(len(target_data)):
        plot_data[i,0:torch.numel(target_data[i])] = target_data[i].cpu().flatten().numpy()
        plot_labels[i] = 0
    for i in range(len(decoy_data)):
        plot_data[i,0:torch.numel(target_data[i])] = target_data[i].cpu().flatten().numpy()
        plot_labels[i] = 1

    mapper = umap.UMAP().fit(plot_data)
    plot.points(mapper, labels=plot_labels)
    plt.savefig(path)

if __name__ == "__main__":
    # define MLP classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.get_num_threads())
    encoder = ProtBertClsEncoder(device)

    # define MLP classifier
    base = 'UP000002311_559292'
    temp_encoding_dir = f"data/encodings/temp_mlp"

    # target data:
    target_file = f"data/targets/{base}.fasta"
    target_records = read_fasta_file(target_file)
    target_sequences = [record.sequence for record in target_records]
    N = 100#len(target_sequences)
    target_lmdb_path = f"{temp_encoding_dir}/targets.lmdb"
    target_labels = torch.zeros(N)
    encode_seqs_to_lmdb(target_sequences[0:N], encoder, target_lmdb_path)

    decoy_files = [f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.reverse.fasta',
                   f'data/decoys/{base}.diann_C.fasta', f'data/decoys/{base}.random_replace.0.fasta',
                   f'data/decoys/{base}.esm8M.best.c1.0.fasta', f'data/decoys/{base}.esm35M.best.c1.0.fasta',
                   f'data/decoys/{base}.esm150M.best.c1.0.fasta', f'data/decoys/{base}.esm650M.best.c1.0.fasta',
                   f'data/decoys/{base}.esm3B.best.c1.16b.0.fasta']
    decoy_ids = ['shuffle', 'reverse', 'diann', 'random_replace', 'esm8M', 'esm35M', 'esm150M', 'esm650M', 'esm3B_16b']
    
    print("Generating UMAP images...")
    for i, decoy_file in enumerate(decoy_files):
        decoy_records = read_fasta_file(decoy_file)
        decoy_sequences = [record.sequence for record in decoy_records]
        M = 100#len(decoy_sequences)
        decoy_lmdb_path = f"{temp_encoding_dir}/{decoy_ids[i]}.lmdb"
        encode_seqs_to_lmdb(decoy_sequences[0:M], encoder, decoy_lmdb_path)
        decoy_labels = torch.ones(M)

        # generate image:
        generate_umap_image(LMDBDataset([target_lmdb_path], target_labels), LMDBDataset([decoy_lmdb_path], decoy_labels),
                            f"src/visualization/images/umap/umap_target_v_{decoy_ids[i]}.png")
    delete_lmdb(target_lmdb_path)

