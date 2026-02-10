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
from src.run.cross_val_rnn import cross_val_rnn
from src.run.cross_val_svm import cross_val_svm
from src.run.generate import generate_decoys
from src.run.timing_test import timing_test
from src.run.fine_tune import fine_tune
from src.io.utils import seed_all
from src.cli.option_lists import get_path, PARAM_PRECISION_TO_TYPE

def process_args(args: argparse.Namespace):
    command = args.command
    seed_all(args.seed)
    if command == "evaluate":
        process_evaluate(args.classifier, args.encoder_model, args.target_file, args.decoy_files, args.decoy_ids)
    elif command == "generate":
        process_generate(args.generator, args.target_file, args.gen_count, args.output_directory, args.seed, args.mask_count,
                         args.parameter_count, args.parameter_precision, args.tuned_model_path)
    elif command == "time":
        process_timing(args.generator, args.target_file, args.timing_sample, args.seed, args.mask_count,
                       args.parameter_count, args.parameter_precision, args.tuned_model_path)
    elif command == "tune":
        process_tune(args.generator, args.training_files, args.output_directory, args.num_epochs, args.batch_size, args.seed, args.mask_count,
                     args.parameter_count, args.parameter_precision, args.tuned_model_path)

def process_evaluate(classifier: str, encoder_model: str, target_file: str, decoy_files: str, decoy_ids: str):
    if classifier == "mlp" and encoder_model == "protbert":
        cross_val_mlp_protbert(target_file, decoy_files, decoy_ids)
    elif classifier == "mlp" and encoder_model == "esm":
        cross_val_mlp_esm(target_file, decoy_files, decoy_ids)
    elif classifier == "rnn":
        cross_val_rnn(target_file, decoy_files, decoy_ids)
    elif classifier == "svm":
        cross_val_svm(target_file, decoy_files, decoy_ids)

def process_generate(generator_string: str, target_file: str, n: int, output_dir: str, seed: int, mask_count: int, 
                     param_count: str, param_precision: int, tuned_model_path: str):
    generator = create_generator_from_parameters(generator_string, seed, mask_count, param_count, param_precision, 
                                                 tuned_model_path)
    generate_decoys(target_file, generator, n, output_dir)

def process_timing(generator_string: str, target_file: str, number_of_seqs_for_timing: int, seed:int, mask_count: int, 
                   param_count: str, param_precision: int, tuned_model_path: str):
    generator = create_generator_from_parameters(generator_string, seed, mask_count, param_count, param_precision, 
                                                 tuned_model_path, "cpu")
    timing_test(target_file, number_of_seqs_for_timing, generator)

def process_tune(generator_string: str, training_files: List[str], model_save_dir: str, num_epochs: int, batch_size: int,
                 seed: int, mask_count: int, param_count: str, param_precision: int, tuned_model_path: str):
    generator = create_generator_from_parameters(generator_string, seed, mask_count, param_count, param_precision, tuned_model_path)
    fine_tune(generator, training_files, model_save_dir, num_epochs, batch_size)

def create_generator_from_parameters(generator_string: str, seed: int, mask_count: int, 
                                     param_count: str, param_precision: int, tuned_model_path: str, device: torch.device = None):
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random: Random = Random(seed)
    special_amino_acids: List[str] = ['R', 'K']
    if generator_string == "shuffle":
        generator = ShuffleGenerator(special_amino_acids=special_amino_acids, random=random)
    elif generator_string == "reverse":
        generator = ReverseGenerator(special_amino_acids=special_amino_acids)
    elif generator_string == "diann":
        generator = DiannGenerator(special_amino_acids=special_amino_acids)
    elif generator_string == "random_replace":
        generator_string = RandomReplaceGenerator(special_amino_acids=special_amino_acids, random=random)
    elif generator_string == "esm":
        local_path = get_path(generator_string, param_count, tuned_model_path)
        weight_type = PARAM_PRECISION_TO_TYPE[param_precision]
        generator = EsmGenerator(
            local_path=local_path,
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=mask_count,
            weight_type=weight_type
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
    elif generator_string == "protbert":
        local_path = get_path(generator_string, param_count, tuned_model_path)
        weight_type = PARAM_PRECISION_TO_TYPE[param_precision]
        generator = ProtBertGenerator(
            local_path=local_path,
            random=random,
            special_amino_acids=special_amino_acids,
            sort_optimization=True,
            batch_size=1,
            ml_generator_type=MlGeneratorType.BEST,
            device=device,
            masking_type=MaskingType.COUNT,
            mask_count=mask_count,
            weight_type=weight_type
        )
    return generator