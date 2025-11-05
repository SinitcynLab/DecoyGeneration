import torch
import numpy as np

from sklearn.model_selection import train_test_split
from src.peptide_classifiers.nn_classifier import train_nn
from src.peptide_classifiers.feed_forward_nn_classifier import FeedForwardNNClassifier
from src.peptide_classifiers.recurrent_nn_classifier import RecurrentNNClassifier
from src.encoders.image_encoder import ImageEncoder
from src.encoders.protbert_pca_encoder import ProtBertPcaEncoder
from src.encoders.protbert_encoder import ProtBertEncoder
from src.io.fasta import read_fasta_file

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # define MLP classifier
    tokenized_len = 64
    pca_dim = 100
    mlp_net = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(pca_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    pca_encoder = ProtBertPcaEncoder(max_tokenized_length=tokenized_len, device=device, pca_dim=pca_dim)
    mlp = FeedForwardNNClassifier(network=mlp_net, encoder=pca_encoder, device=device)

    # define CNN classifier
    # cnn_net = torch.nn.Sequential(
    #     torch.nn.Conv2d(1, 4, kernel_size=3),
    #     torch.nn.MaxPool2d(kernel_size=2),
    #     torch.nn.Conv2d(4, 16, kernel_size=4),
    #     torch.nn.MaxPool2d(kernel_size=2),
    #     torch.nn.Flatten(),
    #     torch.nn.Linear(16*62*254, 128),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(128, 32),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(32, 1),
    #     torch.nn.Sigmoid()
    # )
    # image_encoder = ImageEncoder(image_height=256, device=device)
    # cnn = FeedForwardNNClassifier(network=cnn_net, encoder=image_encoder, device=device)

    # define RNN classifier
    # out_size = 128
    # rnn_module = torch.nn.RNN(1024, out_size)
    # rnn_net = torch.nn.Sequential( # each character (amino acid) is encoded using 1024 numbers
    #     torch.nn.Linear(out_size, 1),
    #     torch.nn.Sigmoid()
    # )
    # var_len_encoder = ProtBertEncoder(device=device, constant_length=False, flatten=False)
    # rnn = RecurrentNNClassifier(rnn=rnn_module, network=rnn_net, encoder=var_len_encoder, device=device)

    extensions = ['esm.best.0']
    nets = [mlp]
    names = ["mlp"] # should probably add name property to classifiers
    base = 'UP000000625_83333'

    for ext in extensions:
        # load data:
        target_records = read_fasta_file("data/targets/UP000002311_559292.fasta")
        decoy_records = read_fasta_file(f"/home/ctrl/DecoyGeneration/data/decoys/{base}.{ext}.fasta")

        if ext == 'targets':
            sequences = [record.sequence for record in target_records]
            target_sequences = sequences[0:len(sequences)//2]
            decoy_sequences = sequences[len(sequences)//2:len(sequences)]
        else:
            target_sequences = [record.sequence for record in target_records]
            decoy_sequences = [record.sequence for record in decoy_records]
            target_records = target_sequences[0:len(decoy_sequences)]

        target_labels = [0 for _ in range(len(target_sequences))]
        decoy_labels = [1 for _ in range(len(decoy_sequences))]

        sequences = target_sequences + decoy_sequences
        labels = target_labels + decoy_labels
        
        test_fraction = 0.3
        X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=test_fraction, random_state=0)
        
        for i, net in enumerate(nets):
            # train net:
            N = 500 # 1000 each
            M = round(N * test_fraction)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
            n_epochs = 20
            batch_size = 10
            best_acc = train_nn(net, X_train[0:N], y_train[0:N], X_val[0:M], y_val[0:M], n_epochs, batch_size, optimizer)
            print(f"Best accuracy of {names[i]} on {ext}: {best_acc}")