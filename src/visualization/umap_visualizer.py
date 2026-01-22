import torch
import numpy as np
import umap
import pandas as pd
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from typing import Iterable

from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.encoders.spectrum_encoder import VectorSpectrumEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset

def generate_umap_image(files: Iterable[str], seq_ids: Iterable[str], number: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ProtBertClsEncoder(device)
    print("Generating UMAP image...")
    data_list = []
    label_list = []
    for i, file in enumerate(files):
        records = read_fasta_file(file)
        sequences = [record.sequence for record in records]
        encodings = encoder(sequences[0:number])
        file_data: np.ndarray = encodings.numpy()
        file_labels = np.ones(N, dtype=int) * i
        
        data_list.append(file_data)
        label_list.append(file_labels)
        print(f"{i+1}/{len(files)}")
    plot_data = np.concat(data_list, axis=0)
    labels = np.concat(label_list, axis=0)
    embedding = umap.UMAP().fit_transform(X=plot_data, y=labels)

    plt.style.use("bmh") # professional look
    size_val = 1
    cmap = plt.get_cmap("Set1")

    for file_suffix in np.unique(labels):
        indices_with_label = np.where(labels == file_suffix)
        data_to_plot = embedding[indices_with_label]
        plt.scatter(data_to_plot[:,0], data_to_plot[:,1], s=size_val, c=cmap(file_suffix))

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.gca().legend(seq_ids, markerscale=5//size_val)
    plt.subplots_adjust(top=0.83)
    plt.title(f"UMAP visualization of target sequences, sequences generated \n using the shuffle procedure and sequences \n generator using ESM 650M")
    file_suffix = "_".join(seq_ids)
    plt.savefig(f"src/visualization/images/umap/umap_{file_suffix}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", help="What files to obtain sequences from to plot in a UMAP image.")
    parser.add_argument("-i", "--identifiers", help="What labels to use for the sequences from the various files.")
    parser.add_argument("-n", "--number", "The number of points to plot per class in the UMAP image.")

    args = parser.parse_args()

    generate_umap_image(args.files, args.identifiers, args.number)