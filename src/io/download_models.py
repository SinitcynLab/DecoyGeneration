from transformers import AutoModel, AutoTokenizer

hugging_face_ids = ["Rostlab/prot_bert", "facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D",
             "facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D", "facebook/esm2_t36_3B_UR50D"]
models_dir = "models"

for hugging_face_model_id in hugging_face_ids:
    # determine name destination folder (relative path):
    model_name = hugging_face_model_id.split("/")[-1]
    specific_model_dir = models_dir + "/" + model_name
    # load the model from huggingface:
    tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_id)
    model = AutoModel.from_pretrained(hugging_face_model_id)
    # actually save the model in the specified directory:
    tokenizer.save_pretrained(specific_model_dir)
    model.save_pretrained(specific_model_dir)
