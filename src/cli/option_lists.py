import torch

CLASSIFIER_LIST = ["mlp", "rnn", "svm"]
GENERATOR_LIST = ["reverse", "shuffle", "diann", "esm650M_32bit", "esm650M_16bit", "protbert_32bit", "protbert_16bit",
                  "esm8M_32bit", "esm8M_16bit", "max_prob_smart_esm", "rel_diff_smart_esm", "random_replace",
                  "esm_c_terminus", "esm_n_terminus"]
COMMAND_LIST = ["evaluate", "generate", "time"]

PARAM_COUNT_TO_PATH: dict = {"8M": "models/esm2_t6_8M_UR50D",
                             "35M": "models/esm2_t12_35M_UR50D",
                             "150M": "models/esm2_t30_150M_UR50D",
                             "650M": "models/esm2_t33_650M_UR50D",
                             "3B": "models/esm2_t36_3B_UR50D",
                             "15B": "esm2_t48_15B_UR50D"}
PARAM_PRECISION_TO_TYPE: dict = {16: torch.float16,
                                 32: torch.float32}