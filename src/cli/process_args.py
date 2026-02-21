import argparse
import torch

from typing import List
from random import Random

from src.decoy_generators.decoy_generator import DecoyGenerator
from src.decoy_generators.diann_generator import DiannGenerator
from src.decoy_generators.esm_generator import EsmGenerator, MaskingType, MlGeneratorType
from src.decoy_generators.terminus_esm_generator import TerminusEsmGenerator
from src.decoy_generators.reverse_generator import ReverseGenerator
from src.decoy_generators.shuffle_generator import ShuffleGenerator
from src.decoy_generators.protbert_generator import ProtBertGenerator
from src.decoy_generators.random_replace_generator import RandomReplaceGenerator
from src.run.cross_val_mlp import cross_val_mlp_protbert, cross_val_mlp_esm
from src.run.cross_val_rnn import cross_val_rnn_protbert, cross_val_rnn_esm
from src.run.cross_val_plm_free import cross_val_plm_free
from src.run.cross_val_svm import cross_val_svm
from src.run.generate import generate_decoys
from src.run.timing_test import timing_test
from src.io.utils import seed_all
from src.cli.option_lists import get_model_name, PARAM_PRECISION_TO_TYPE

from src.proteins.protease import get_protease


def process_args(args: argparse.Namespace):
    command = args.command
    seed_all(args.seed)
    if command == "evaluate":
        process_evaluate(args)
    elif command == "generate":
        process_generate(args)
    elif command == "time":
        process_timing(args)

def process_evaluate(args: argparse.Namespace):
    if args.classifier == "mlp" and args.encoder_model == "protbert":
        cross_val_mlp_protbert(args.target_file, args.decoy_files, args.decoy_ids)
    elif args.classifier == "mlp" and args.encoder_model == "esm":
        cross_val_mlp_esm(args.target_file, args.decoy_files, args.decoy_ids)
    elif args.classifier == "rnn" and args.encoder_model == "protbert":
        cross_val_rnn_protbert(args.target_file, args.decoy_files, args.decoy_ids)
    elif args.classifier == "rnn" and args.encoder_model == "esm":
        cross_val_rnn_esm(args.target_file, args.decoy_files, args.decoy_ids)
    elif args.classifier == "svm":
        protease = get_protease(args.protease)
        cross_val_svm(args.target_file, args.decoy_files, args.decoy_ids, protease)
    elif args.classifier == "plm_free":
        protease = get_protease(args.protease)
        cross_val_plm_free(args.target_file, args.decoy_files, args.decoy_ids, protease)

def process_generate(args: argparse.Namespace):
    generator = create_generator_from_parameters(args)
    generate_decoys(args.target_file, generator, args.gen_count, args.output_directory)

def process_timing(args: argparse.Namespace):
    generator = create_generator_from_parameters(args, device="cpu")
    timing_test(args.target_file, args.timing_sample, generator)

def create_generator_from_parameters(args: argparse.Namespace, device: torch.device = None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random: Random = Random(args.seed)
    protease = get_protease(args.protease)
    if args.generator == "shuffle":
        generator = ShuffleGenerator(protease=protease, random=random)
    elif args.generator == "reverse":
        generator = ReverseGenerator(protease=protease)
    elif args.generator == "diann":
        generator = DiannGenerator(protease=protease)
    elif args.generator == "random_replace":
        generator = RandomReplaceGenerator(protease=protease, random=random)
    elif args.generator == "esm":
        model_name = get_model_name(model_type=args.generator, model_size=args.parameter_count, custom_model_path=args.tuned_model_path)
        dtype = PARAM_PRECISION_TO_TYPE[args.parameter_precision]
        generator = EsmGenerator(
            model_name=model_name,
            random=random,
            protease=protease,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=args.mask_count,
            dtype=dtype
        )
    elif args.generator == "esm_n_terminus":
        model_name = get_model_name(model_type="esm", model_size=args.parameter_count, custom_model_path=args.tuned_model_path)
        dtype = PARAM_PRECISION_TO_TYPE[args.parameter_precision]
        generator = TerminusEsmGenerator(
            model_name=model_name,
            random=random,
            protease=protease,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=args.mask_count,
            terminus='N'
        )
    elif args.generator == "esm_c_terminus":
        model_name = get_model_name(model_type="esm", model_size=args.parameter_count, custom_model_path=args.tuned_model_path)
        dtype = PARAM_PRECISION_TO_TYPE[args.parameter_precision]
        generator = TerminusEsmGenerator(
            model_name=model_name,
            random=random,
            protease=protease,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=args.mask_count,
            terminus='C'
        )
    elif args.generator == "protbert":
        model_name = get_model_name(model_type="protbert", model_size=args.parameter_count, custom_model_path=args.tuned_model_path)
        dtype = PARAM_PRECISION_TO_TYPE[args.parameter_precision]
        generator = ProtBertGenerator(
            model_name=model_name,
            random=random,
            protease=protease,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=args.mask_count,
            dtype=dtype,
        )

    return generator