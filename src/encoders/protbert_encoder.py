import torch

from src.encoders.transformer_encoder import TransformerEncoder
from typing import Iterable, List
from transformers import BertModel, BertTokenizer

class ProtBertEncoder(TransformerEncoder):
    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert",
        max_tokenized_length: int = 64,
        device='cpu',
        constant_length: bool = True,
        flatten: bool = True
    ):
        TransformerEncoder.__init__(self, max_tokenized_length, device, constant_length, flatten)

        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        self.model.eval()
        self.model.to(self.device)
        self.flatten = flatten

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        sequences: List[str] = [" ".join(sequence) for sequence in sequences]
        return TransformerEncoder.__call__(self, sequences)
