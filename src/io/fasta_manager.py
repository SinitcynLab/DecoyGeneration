import argparse

from src.io.combiner import Combiner
from src.io.relabler import Relabler

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", help="Determine command to be executed (either 'combine' or 'relabel').")
    parser.add_argument("-i", "--input_files", nargs="+",
                        help="Input files to execute command on. (If multiple files, provide names separated by spaces, e.g. 'a b c'.)")
    parser.add_argument("-o", "--output_files", nargs="+",
                        help="Output files to write into. (If multiple files, provide names separated by spaces, e.g. 'a b c'.)")
    parser.add_argument("-l", "--label", help="Label to prepend to fasta files in case command is relable.")

    return parser.parse_args()

def main():
    args = collect_args()
    i_files = args.input_files
    o_files = args.output_files
    if args.command == 'combine':
        if len(o_files) > 1:
            raise ValueError("Combine can only write to single file.")
        else:
            o_file = o_files[0]
        Combiner.combine(i_files, o_file)
    elif args.command == 'relabel':
        if not(args.label):
            raise ValueError("To relabel, you must pass the --label (-l) argument.")
        else:
            label = args.label
        if len(i_files) != len(o_files):
            raise ValueError("relabel only accepts input and output lists of equal length, and relabels input to output files respectively.")
        file_pairs = zip(i_files, o_files)
        for pair in file_pairs:
            Relabler.relabel(label, pair[0], pair[1])
    print("Operation succesful.")

if __name__=="__main__":
    main()