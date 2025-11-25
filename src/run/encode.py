import argparse
import torch

from src.io.fasta import read_fasta_file
from src.encoders.protbert_encoder import ProtBertEncoder

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="File with proteins to encode.")
    parser.add_argument("-e", "--encoder", help="Encoder to use")
    parser.add_argument("-l", "--length", help="Length of encoding to use.")
    parser.add_argument("-o", "--output_file", help="File to which tensors must be written")

    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = collect_args()

    i_file_name = args.input_file
    fasta_records = read_fasta_file(i_file_name)
    sequences = [record.sequence for record in fasta_records]

    o_file_name = args.output_file

    if args.encoder == 'recurrent':
        encoder = ProtBertEncoder(device=device, constant_length=False, flatten=False)
        data = encoder(sequences)
    else:
        raise ValueError("Specify a valid encoder.")

    torch.save(data, file=o_file_name)

if __name__=="__main__":
    main()