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
from src.decoy_generators.smart_masking_esm import MaxProbMaskingEsmGenerator, RelDiffMaskingEsmGenerator, MaxEntropyEsmGenerator
from src.decoy_generators.random_replace_generator import RandomReplaceGenerator
from src.run.cross_val_mlp import cross_val_mlp
from src.run.cross_val_rnn import cross_val_rnn
from src.run.cross_val_svm import cross_val_svm
from src.run.generate import generate_decoys
from src.run.timing_test import timing_test
from src.io.utils import seed_all

def process_args(args: argparse.Namespace):
    command = args.command
    seed_all(args.seed)
    if command == "evaluate":
        process_evaluate(args.classifier, args.target_file, args.decoy_files, args.decoy_ids)
    elif command == "generate":
        process_generate(args.generators, args.target_file, args.gen_count, args.output_directory, args.seed, args.mask_count)
    elif command == "time":
        process_timing(args.generators, args.target_file, args.timing_sample)

def process_evaluate(classifier: str, target_file: str, decoy_files: str, decoy_ids: str):
    if classifier == "mlp":
        cross_val_mlp(target_file, decoy_files, decoy_ids)
    elif classifier == "rnn":
        cross_val_rnn(target_file, decoy_files, decoy_ids)
    elif classifier == "svm":
        cross_val_svm(target_file, decoy_files, decoy_ids)

def process_generate(generator_strings: List[str], target_file: str, n: int, output_dir: str, seed: int, mask_count: int):
    generators = create_generators_from_list(generator_strings, seed, mask_count)
    generate_decoys(target_file, generators, n, output_dir)

def process_timing(generator_strings: List[str], target_file: str, number_of_seqs_for_timing: int, seed:int, mask_count: int):
    generators = create_generators_from_list(generator_strings, seed, mask_count, "cpu")
    timing_test(target_file, number_of_seqs_for_timing, generators)

def create_generators_from_list(generator_strings: List[str], seed: int, mask_count: int, device: torch.device = None):
    generators: List[DecoyGenerator] = []
    for generator_string in generator_strings:
        generators.append(create_generator_from_string(generator_string, seed, mask_count, device))
    return generators

def create_generator_from_string(generator_string: str, seed: int, mask_count: int, device: torch.device = None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random: Random = Random(seed)
    special_amino_acids: List[str] = ['R', 'K']

    # TODO: decompose the specification of generators from single strings to multiple separate arguments
    # e.g. instead of specifying 'esm650M_32bit', you specify 'esm', '650' and '32' as separate arguments. That would severly cut down the
    # size of the if-statement below.
    if generator_string == "shuffle":
        generator = ShuffleGenerator(special_amino_acids=special_amino_acids, random=random)
    elif generator_string == "reverse":
        generator = ReverseGenerator(special_amino_acids=special_amino_acids)
    elif generator_string == "diann":
        generator = DiannGenerator(special_amino_acids=special_amino_acids)
    elif generator_string == "random_replace":
        generator_string = RandomReplaceGenerator(special_amino_acids=special_amino_acids, random=random)
    elif generator_string == "esm8M_32bit":
        generator = EsmGenerator(
            local_path="models/esm2_t6_8M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=mask_count
        )
    elif generator_string == "esm8M_16bit":
        generator = EsmGenerator(
            local_path="models/esm2_t6_8M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=mask_count,
            weight_type=torch.float16
        )
    elif generator_string == "esm650M_32bit":
        generator = EsmGenerator(
            local_path="models/esm2_t33_650M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=mask_count
        )
    elif generator_string == "esm650M_16bit":
        generator = EsmGenerator(
            local_path="models/esm2_t33_650M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=mask_count,
            weight_type=torch.float16
        )
    elif generator_string == "max_prob_smart_esm":
        generator = MaxProbMaskingEsmGenerator(
            local_path="models/esm2_t6_8M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
        )
    elif generator_string == "rel_diff_smart_esm":
        generator = RelDiffMaskingEsmGenerator(
            local_path="models/esm2_t6_8M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
        )
    elif generator_string == "esm_n_terminus":
        generator = TerminusEsmGenerator(
            local_path="models/esm2_t33_650M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=mask_count,
            terminus='N'
        )
    elif generator_string == "esm_c_terminus":
        generator = TerminusEsmGenerator(
            local_path="models/esm2_t33_650M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=mask_count,
            terminus='C'
        )
    elif generator_string == "max_entropy_esm":
        generator = MaxEntropyEsmGenerator(
            local_path="models/esm2_t6_8M_UR50D",
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
        )
    return generator