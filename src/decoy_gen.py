import argparse

from src.io.combiner import Combiner
from src.io.relabler import Relabler

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", help="Determine command to be executed ('evaluate', 'generate' or 'time').")
    parser.add_argument("--decoy_files", nargs="+",
                        help="Decoy files to execute command on.")
    parser.add_argument("--decoy_ids", nargs="+", 
                        help="String names for the provided decoy files.")
    parser.add_argument("--generators", nargs="+",
                        help="Generators to execute command with. (Choose from 'reverse', 'shuffle', 'diann', 'esm650M_32bit', 'esm650M_16bit'," + 
                        " 'esm8M_32bit', 'esm8M_16bit', 'smart_masking_esm'.)")
    parser.add_argument("--output_directory", help="Output directory to write into.")
    parser.add_argument("--target_file", help="Target file to execute command on.")
    parser.add_argument("--classifier", help="The classifier to execute the command with. (Choose from 'mlp', 'rnn').")

    return parser.parse_args()

def main():
    args = collect_args()
    i_files = args.input_files
    o_files = args.output_files
    


if __name__=="__main__":
    main()