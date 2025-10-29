import torch
from typing import Iterable

from src.encoders.peptide_encoder import PeptideEncoder

class TransformerEncoder(PeptideEncoder):
    def __init__(self, max_tokenized_length : int = 256, device : torch.device = 'cpu'):
        PeptideEncoder.__init__(self)
        self.max_tokenized_length = max_tokenized_length
        self.device = device
    
    def _embed_batched(self, sequences : Iterable[str], batch_size : int = 32):
        batch_starts = torch.arange(0, len(sequences), batch_size)
        output_list = []
        for batch_start in batch_starts:
            batch_end = min(batch_start + batch_size, len(sequences)-1)
            batch_sequences = sequences[batch_start:batch_end]
            batch_inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True, max_length = self.max_tokenized_length, truncation=True)
            batch_inputs.to(self.device)
            with torch.no_grad():
                batch_masked_lm_outputs = self.model(**batch_inputs, output_hidden_states=True)
            batch_hidden_states = batch_masked_lm_outputs.hidden_states[0] # 0-th hidden states are the embeddings
            output_list.append(batch_hidden_states)
        return torch.cat(output_list, axis=0)