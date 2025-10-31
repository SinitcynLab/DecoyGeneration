import torch
from typing import Iterable

from transformers import EsmTokenizer, EsmForMaskedLM

from src.encoders.peptide_encoder import PeptideEncoder

class TransformerEncoder(PeptideEncoder):
    def __init__(self, max_tokenized_length : int = 256, device : torch.device = 'cpu'):
        PeptideEncoder.__init__(self)
        self.max_tokenized_length = max_tokenized_length
        self.device = device
        self.canonical_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    def _load_transformer_from_path(self, path : str):
        model = EsmForMaskedLM.from_pretrained(path, local_files_only=True)
        tokenizer = EsmTokenizer.from_pretrained(path, local_files_only=True)
        tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        model.eval()
        model.to(self.device)
        return model, tokenizer

    def _embed_batched(self, sequences : Iterable[str], batch_size : int = 32):
        batch_starts = torch.arange(0, len(sequences), batch_size)
        output_list = []
        for batch_start in batch_starts:
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            batch_inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding='max_length', max_length=self.max_tokenized_length, truncation=True)
            batch_inputs.to(self.device)
            with torch.no_grad():
                batch_outputs = self.model(**batch_inputs, output_hidden_states=True)
            if hasattr(batch_outputs, 'last_hidden_state'):
                batch_hidden_states = batch_outputs.last_hidden_state # extract the embeddings
            else:
                batch_hidden_states = batch_outputs.hidden_states[-1] # extract the embeddings
                batch_hidden_states = self.normalize_tensor(batch_hidden_states) # normalize (embedding not always normalized in this case)
            output_list.append(batch_hidden_states)
            print("%d / %d" % (batch_end, len(sequences)))
        return torch.cat(output_list, axis=0)