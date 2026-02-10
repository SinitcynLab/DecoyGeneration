import argparse

from src.cli.process_args import process_args
from src.cli.validate_args import validate_args
from src.cli.option_lists import CLASSIFIER_LIST, COMMAND_LIST, GENERATOR_LIST, ENCODER_LIST

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", help=f"Determine command to be executed (choose from {COMMAND_LIST}).")
    parser.add_argument("--target_file", help="The file holding the target sequences that you want to execute the command on.")
    
    parser.add_argument("--classifier", help=f"The classifier to execute the command with. (Choose from {CLASSIFIER_LIST}).")
    parser.add_argument("--decoy_files", nargs="+",
                        help="Decoy files to execute command on.")
    parser.add_argument("--decoy_ids", nargs="+", 
                        help="String names for the provided decoy files.")
    parser.add_argument("--encoder_model", type=str, default="protbert", 
                        help=f"Which model to use for encoding sequences before classification (defaults to protbert). (Choose from {ENCODER_LIST}).")
    
    parser.add_argument("--generator", help=f"Generator to execute command with. (Choose from {GENERATOR_LIST}.)")
    parser.add_argument("--parameter_count", type=str, default="650M", 
                        help="The number of parameters to use when ESM is used as a generator (defaults to 650M).")
    parser.add_argument("--parameter_precision", type=int, default=32, help="The precision to use for the parameters of the generator in bits (defaults to 32).")
    parser.add_argument("--output_directory", help="Output directory to write into.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use for generation (defaults to 42).")
    parser.add_argument("--mask_count", type=int, default=1, help="The masking count to use for ESM generators (defaults to 1).")
    parser.add_argument("--gen_count", type=int, default=1, help="The number of decoy files to generate if a decoy method is one-to-many (defaults to 1).")
    parser.add_argument("--tuned_model_path", type=str, 
                        help="The path to the fine-tuned model you would like to  use for generation.")

    parser.add_argument("--timing_sample", type=int, default=100, help="The number of sequences from the target file to use for timing measurements, taken from the start of the file (defaults to 100).")

    parser.add_argument("--training_files", nargs="+",
                        help="Fasta files to use for fine-tuning the chosen generator.")
    parser.add_argument("--num_epochs", type=int, default=3, help="The number of epochs used for fine tuning the model for (defaults to 3).")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size used for fine tuning (defaults to 8).")

    return parser.parse_args()

if __name__=="__main__":
    args = collect_args()
    validate_args(args)
    process_args(args)