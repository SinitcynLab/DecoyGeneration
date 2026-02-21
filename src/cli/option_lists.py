import torch

CLASSIFIER_LIST = ["mlp", "rnn", "svm", "plm_free"]
ENCODER_LIST = ["esm", "protbert"]
GENERATOR_LIST = ["reverse", "shuffle", "diann", "esm", "protbert", "protbert_32bit", "protbert_16bit",
                  "esm_c_terminus", "esm_n_terminus", "random_replace"]
PARAMETER_COUNT_LIST = ["8M", "35M", "150M", "650M", "3B"]
PARAMETER_PRECISION_LIST = [16, 32]
COMMAND_LIST = ["evaluate", "generate", "time"]

PARAM_PRECISION_TO_TYPE: dict = {
    16: torch.float16,
    32: torch.float32,
}

def get_model_name(model_type: str, model_size: str, custom_model_path: str | None = None) -> str:
    if custom_model_path != None:
        return custom_model_path
    elif model_type == "protbert":
        return "Rostlab/prot_bert"
    elif model_type == "esm":
        model_size_to_model_name: dict = {
            "8M": "facebook/esm2_t6_8M_UR50D",
            "35M": "facebook/esm2_t12_35M_UR50D",
            "150M": "facebook/esm2_t30_150M_UR50D",
            "650M": "facebook/esm2_t33_650M_UR50D",
            "3B": "facebook/esm2_t36_3B_UR50D",
            "15B": "facebook/esm2_t48_15B_UR50D",
        }
        return model_size_to_model_name[model_size]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
