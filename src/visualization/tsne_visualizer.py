import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

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

    files = [f'data/targets/{base}.fasta', f'data/decoys/{base}.shuffle.0.fasta']
    seq_ids = ('targets', 'shuffle')
    
    print("Generating t-SNE image...")
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

    X_tsne = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(plot_data)
    for i, label in enumerate(seq_ids):
        idx = np.where(labels == i)
        plt.scatter(X_tsne[idx,0], X_tsne[idx,1], label=label)
    plt.title(f"t-SNE visualization of target vs decoy sequences (Protbert CLS encoder)")
    plt.legend()
    plt.savefig("src/visualization/images/umap/tsne_total_3_class.png")