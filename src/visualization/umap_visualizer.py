import torch
import numpy as np
import umap
import pandas as pd

from umap import plot
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.encoders.spectrum_encoder import VectorSpectrumEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset

def generate_umap_image(targets: LMDBDataset, decoys: LMDBDataset, path: str, decoy_type: str):
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
    plt.legend(('Targets', f'Decoys ({decoy_type})'))
    plt.title(f"UMAP visualization of target vs decoy sequences ({decoy_type})")
    plt.savefig(path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.get_num_threads())
    encoder = ProtBertClsEncoder(device)

    # Choose base file:
    base = 'UP000002311_559292'
    files = [f'data/targets/{base}.fasta', f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.esm650M.best.c1.0.fasta']
    seq_ids = ('Target', 'Shuffle', 'ESM 650M')
    
    print("Generating UMAP image...")
    data_list = []
    label_list = []
    for i, file in enumerate(files):
        records = read_fasta_file(file)
        sequences = [record.sequence for record in records]
        N = 100#len(sequences)
        encodings = encoder(sequences[0:N])
        file_data: np.ndarray = encodings.numpy()
        file_labels = np.ones(N, dtype=int) * i
        
        data_list.append(file_data)
        label_list.append(file_labels)
        print(f"{i+1}/{len(files)}")
    plot_data = np.concat(data_list, axis=0)
    labels = np.concat(label_list, axis=0)
    embedding = umap.UMAP().fit_transform(X=plot_data, y=labels)

    plt.style.use("bmh") # professional look

    for label in np.unique(labels):
        indices_with_label = np.where(labels == label)
        data_to_plot = embedding[indices_with_label]
        plt.scatter(data_to_plot[:,0], data_to_plot[:,1], s=1)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.gca().legend(seq_ids)
    plt.title(f"UMAP visualization of target sequences, sequences generated \n using the shuffle procedure and sequences \n generator using ESM 650M")
    plt.savefig("src/visualization/images/umap/umap_publish_target_v_shuffle.png")