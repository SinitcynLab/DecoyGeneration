import torch
import numpy as np
import umap
from umap import plot
import matplotlib.pyplot as plt

from src.encoders.spectrum_encoder import VectorSpectrumEncoder
from src.io.fasta import read_fasta_file

if __name__ == "__main__":
    # define MLP classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.get_num_threads())
    special_amino_acids = ['L', 'R']
    encoder = VectorSpectrumEncoder(special_amino_acids)

    # Choose base file:
    base = 'UP000002311_559292'

    files = [f'data/targets/{base}.fasta', f'data/targets/{base}.fasta']
    seq_ids = ('targets', 'diann', 'esm650M')
    
    print("Generating UMAP image...")
    data_list = []
    label_list = []
    for i, file in enumerate(files):
        records = read_fasta_file(file)
        sequences = [record.sequence for record in records]
        N = 3000
        encodings = encoder(sequences[N*i:N*(i+1)])
        K = len(encodings)
        encodings = torch.cat(encodings)
        file_data: np.ndarray = encodings.numpy()
        file_labels = np.ones(K, dtype=int) * i
        
        data_list.append(file_data)
        label_list.append(file_labels)
        print(f"{i+1}/{len(files)}")
    plot_data = np.concat(data_list, axis=0)
    labels = np.concat(label_list, axis=0)
    mapper = umap.UMAP().fit(X=plot_data)
    plot.points(mapper, labels=labels, color_key_cmap='Paired')
    #plt.gca().legend(seq_ids)
    plt.title(f"UMAP visualization of target vs decoy sequences (Protbert CLS encoder)")
    plt.savefig("src/visualization/images/umap/umap_spectra(targets vs targets).png")