import argparse

from src.cli.process_args import process_args
from src.cli.validate_args import validate_args

CLASSIFIER_LIST = ["mlp", "rnn"]
GENERATOR_LIST = ["reverse", "shuffle", "diann", "esm650M_32bit", "esm650M_16bit", 
                  "esm8M_32bit", "esm8M_16bit", "smart_masking_esm"]
COMMAND_LIST = ["evaluate", "generate", "time"]

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", help=f"Determine command to be executed (choose from {COMMAND_LIST}).")
    parser.add_argument("--target_file", help="The file holding the target sequences that you want to execute the command on.")
    
    parser.add_argument("--classifier", help=f"The classifier to execute the command with. (Choose from {CLASSIFIER_LIST}).")
    parser.add_argument("--decoy_files", nargs="+",
                        help="Decoy files to execute command on.")
    parser.add_argument("--decoy_ids", nargs="+", 
                        help="String names for the provided decoy files.")
    
    parser.add_argument("--generators", nargs="+",
                        help=f"Generators to execute command with. (Choose from {GENERATOR_LIST}.)")
    parser.add_argument("--output_directory", help="Output directory to write into.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use for generation (defaults to 42).")
    parser.add_argument("--mask_count", type=int, default=1, help="The masking count to use for ESM generators (defaults to 1).")
    parser.add_argument("--gen_count", type=int, default=1, help="The number of decoy files to generate if a decoy method is one-to-many (defaults to 1).")

    parser.add_argument("--timing_sample", type=int, default=100, help="The number of sequences from the target file to use for timing measurments, taken from the start of the file (defaults to 100).")

    return parser.parse_args()

if __name__=="__main__":
    args = collect_args()
    validate_args(args)
    process_args(args)