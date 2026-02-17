from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
from typing import List

from src.decoy_generators.ml_generator import MlGenerator
from src.decoy_generators.esm_generator import EsmGenerator
from src.decoy_generators.protbert_generator import ProtBertGenerator
from src.io.fasta import read_fasta_file

MODEL_TYPE_TO_PARAMS: dict = {EsmGenerator: {"learning_rate": 4e-4, "weight_decay": 0.01}, # source: https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf
                         ProtBertGenerator: {"learning_rate": 2e-3, "weight_decay": 0.01}} # source: https://ieeexplore.ieee.org/ielaam/34/9893033/9477085-aam.pdf, https://huggingface.co/Rostlab/prot_bert#model-description

def get_training_arguments(ml_generator: MlGenerator, model_save_dir: str, num_epochs: int, batch_size: int):
    param_dict = MODEL_TYPE_TO_PARAMS[type(ml_generator)]
    log_dir = model_save_dir + "/mlm_results"
    return TrainingArguments(
        output_dir=log_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=param_dict["learning_rate"],
        weight_decay=param_dict["weight_decay"]
    )

def fine_tune(ml_generator: MlGenerator, training_files: List[str], model_save_dir: str, num_epochs: int, batch_size: int):
    # load target data and prep dataset:
    target_sequences = []
    for file in training_files:
        target_sequences = target_sequences + list(read_fasta_file(file))
    data: dict = {"sequences": target_sequences}
    dataset: Dataset = Dataset.from_dict(data)
    
    # define tokenize function using the DecoyGenerator's tokenizer:
    def tokenize_function(samples):
        return ml_generator.tokenizer(
            samples["sequences"]
        )
    
    # created tokenized dataset object using tokenize function:
    tokenized_dataset: Dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sequences"]
    )

    # Create data collator to handle random masking for typical BERT training:
    data_collator: DataCollatorForLanguageModeling = DataCollatorForLanguageModeling(
        tokenizer=ml_generator.tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # define a train-val split (default val 20% of data):
    split: DatasetDict = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset: Dataset = split["train"]
    eval_dataset: Dataset = split["test"]
    
    # determine training args and create trainer:
    training_args = get_training_arguments(ml_generator, model_save_dir, num_epochs, batch_size)
    trainer: Trainer = Trainer(
        model=ml_generator.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=ml_generator.tokenizer,
        data_collator=data_collator
    )

    # actually train the model:
    trainer.train()

    # save the trained model:
    ml_generator.model.save_pretrained(model_save_dir)   
    ml_generator.tokenizer.save_pretrained(model_save_dir)   

    return