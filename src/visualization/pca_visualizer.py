import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from src.peptide_classifiers.nn_classifier import cross_validate_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.encoders.protbert_cls_encoder import ProtBertClsEncoder
from src.encoders.spectrum_encoder import VectorSpectrumEncoder
from src.io.fasta import read_fasta_file
from src.io.lmdb_writer import encode_seqs_to_lmdb, delete_lmdb
from src.io.lmdb_dataset import LMDBDataset


if __name__ == "__main__":
    # define MLP classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.get_num_threads())
    encoder = ProtBertClsEncoder(device)

    # Choose base file:
    base = 'UP000002311_559292'

    files = [f'data/targets/{base}.fasta', f'data/decoys/{base}.shuffle.0.fasta', f'data/decoys/{base}.reverse.fasta',
                   f'data/decoys/{base}.diann_C.fasta', f'data/decoys/{base}.random_replace.0.fasta',
                   f'data/decoys/{base}.esm8M.best.c1.0.fasta', f'data/decoys/{base}.esm35M.best.c1.0.fasta',
                   f'data/decoys/{base}.esm150M.best.c1.0.fasta', f'data/decoys/{base}.esm650M.best.c1.0.fasta',
                   f'data/decoys/{base}.esm3B.best.c1.16b.0.fasta']
    seq_ids = ('targets', 'shuffle', 'reverse', 'diann', 'random_replace', 'esm8M', 'esm35M', 'esm150M', 'esm650M', 'esm3B_16b')
    
    print("Generating PCA image...")
    data_list = []
    label_list = []
    for i, file in enumerate(files):
        records = read_fasta_file(file)
        sequences = [record.sequence for record in records]
        N = 100
        encodings = encoder(sequences[0:N])

        file_data: np.ndarray = encodings.numpy()
        file_labels = torch.ones(N) * i
        
        data_list.append(file_data)
        label_list.append(file_labels)
        print(f"{i+1}/{len(files)}")
    plot_data = np.concat(data_list, axis=0)
    labels = np.concat(label_list, axis=0)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(plot_data)
    for i, label in enumerate(seq_ids):
        idx = np.where(labels == i)
        plt.scatter(X_pca[idx,0], X_pca[idx,1], label=label)
    plt.title(f"UMAP visualization of target vs decoy sequences (Protbert CLS encoder)")
    plt.legend()
    plt.savefig("src/visualization/images/umap/pca_total.png")