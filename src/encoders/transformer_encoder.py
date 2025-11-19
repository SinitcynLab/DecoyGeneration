import gc
import torch
from typing import Iterable, Union

from transformers import EsmTokenizer, EsmForMaskedLM

from src.encoders.peptide_encoder import PeptideEncoder

class TransformerEncoder(PeptideEncoder):
    def __init__(self, max_tokenized_length : int = 256, device : torch.device = 'cpu', constant_length : bool = True, cls_only : bool = False):
        PeptideEncoder.__init__(self)
        self.max_tokenized_length = max_tokenized_length
        self.device = device
        self.canonical_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.constant_length = constant_length
        self.cls_only = cls_only

    def _load_transformer_from_path(self, path : str):
        model = EsmForMaskedLM.from_pretrained(path, local_files_only=True)
        tokenizer = EsmTokenizer.from_pretrained(path, local_files_only=True)
        tokenizer.convert_tokens_to_ids(self.canonical_amino_acids)
        model.eval()
        model.to(self.device)
        return model, tokenizer
    
    def __extract_hidden_state(self, output_object) -> torch.Tensor:
        if hasattr(output_object, 'last_hidden_state'):
            hidden_states = output_object.last_hidden_state # extract the embeddings
        else:
            hidden_states = output_object.hidden_states[-1] # extract the embeddings
        if self.cls_only:
            hidden_states = hidden_states[0][0] # [1, num_tokens, 1024], first token is CLS token
            return hidden_states.unsqueeze(0) # add dimension to get each sample as a row
        else:
            return hidden_states

    def _embed_batched(self, sequences : Iterable[str], batch_size : int = 32) -> Union[torch.Tensor, list[torch.Tensor]]:
        if not self.constant_length:
            batch_size = 1 # if non-constant length, one sample per batch.
        batch_starts = torch.arange(0, len(sequences), batch_size)
        output_list = []
        for batch_start in batch_starts:
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            # Truncate/pad sequences if we mandate constant length, do not otherwise:
            if self.constant_length:
                batch_inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding='max_length', max_length=self.max_tokenized_length, truncation=True)
            else:
                batch_inputs = self.tokenizer(batch_sequences, return_tensors="pt")
            batch_inputs.to(self.device)
            with torch.no_grad():
                batch_outputs = self.model(**batch_inputs, output_hidden_states=True)
            batch_hidden_states = self.__extract_hidden_state(batch_outputs)
            batch_hidden_states.cpu()
            output_list.append(batch_hidden_states)
            torch.cuda.empty_cache()
            del batch_inputs, batch_outputs
            gc.collect()

        # Return output as tensor if we mandate constant length, output list otherwise:
        if self.constant_length or self.cls_only:
            return torch.cat(output_list, axis=0)
        else:
            return output_list