import torch

from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable
from transformers import AutoTokenizer, AutoModel, pipeline

class ProtBertEncoder(PeptideEncoder):
    def __init__(self):
        PeptideEncoder.__init__(self)

    def __call__(self, sequences: Iterable[str]) -> torch.Tensor:
        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = AutoModel.from_pretrained("Rostlab/prot_bert")

        fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
        embedding = fe(sequences)
        embedding_tensor = torch.FloatTensor(embedding).mean(dim=2).squeeze(dim=[1, 2]) # uses mean pooling
        return embedding_tensor