import argparse

from src.cli.option_lists import CLASSIFIER_LIST, COMMAND_LIST, GENERATOR_LIST, PARAMETER_COUNT_LIST, PARAMETER_PRECISION_LIST

from src.proteins.protease import list_proteases


def validate_args(args: argparse.Namespace):
    if args.command == "evaluate":
        if args.classifier not in CLASSIFIER_LIST:
            raise ValueError(f"Choose classifiers from {CLASSIFIER_LIST}.")
        if args.decoy_files is None:
            raise ValueError("Please provide decoy files to classify against")
        if len(args.decoy_files) != len(args.decoy_ids):
            raise ValueError("Please ensure that the list of decoy files is as long as the list of decoy ids.")
    elif args.command == "generate":
        validate_generator(args)
        if args.output_directory is None:
            raise ValueError("Please provide an output directory for the decoy .fasta files.")
    elif args.command == "time":
        validate_generator(args)
    elif args.command not in COMMAND_LIST:
        raise ValueError(f"Please choose a command from {COMMAND_LIST}.")
    if args.protease not in list_proteases():
        raise ValueError(f"Please choose a protease from {list_proteases()}.")

def validate_generator(args: argparse.Namespace):
    if args.generator is None:
        raise ValueError("Please provide a generator which must create the files.")
    if args.generator not in GENERATOR_LIST:
        raise ValueError(f"Choose generators from {GENERATOR_LIST}.")
    if args.parameter_count not in PARAMETER_COUNT_LIST:
        raise ValueError(f"Choose a parameter count for esm from {PARAMETER_COUNT_LIST}.")
    if args.parameter_precision not in PARAMETER_PRECISION_LIST:
        raise ValueError(f"Choose a parameter precision from {PARAMETER_PRECISION_LIST} (measured in bits).")
    if args.generator == "protbert" and args.parameter_precision == 16:
        raise ValueError(f"The protbert generator only supports 32-bit parameter precision.")
    