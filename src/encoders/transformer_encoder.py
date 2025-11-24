import gc
import torch
from typing import Iterable, Tuple, List

from transformers import EsmTokenizer, EsmForMaskedLM
from torch.cuda import memory_usage

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
            return self.__extract_cls_token(hidden_states)
        else:
            return hidden_states
        
    def __extract_cls_token(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states[:,0,:] # [batch_size, num_tokens, 1024], first token is CLS token
        return hidden_states # add dimension to get each sample as a row
    
    def __embed_batch_inputs(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        #batch_inputs = batch_inputs.to(self.device)
        with torch.no_grad():
            batch_outputs = self.model(**batch_inputs, output_hidden_states=True)
        batch_hidden_st_gpu = self.__extract_hidden_state(batch_outputs)
        batch_hidden_st_cpu = batch_hidden_st_gpu.cpu()
        free, total = torch.cuda.mem_get_info(self.device)
        mem_used_MB = (total - free) / 1024 ** 2
        print(mem_used_MB)
        return batch_hidden_st_cpu

    def _embed_batched_varied_length(self, sequences : Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_list = []
        for i in range(len(sequences)):
            batch_sequences = sequences[i:i+1]
            batch_inputs = self.tokenizer(batch_sequences, return_tensors="pt")
            output_list.append(self.__embed_batch_inputs(batch_inputs))
        
        # pad output to all get same length s.t. we can pass it around as a tensor:
        lengths: List[int] = [t.size(dim=1) for t in output_list]
        lengths = torch.IntTensor(lengths)
        max_len: int = torch.max(lengths)
        for i in range(len(output_list)):
            diff = max_len - output_list[i].size(dim=1)
            pad = torch.zeros((1, diff, 1024))
            output_list[i] = torch.cat((output_list[i], pad), axis=1)
        return torch.cat(output_list, axis=0), lengths

    def _embed_batched_constant_length(self, sequences : Iterable[str], batch_size : int = 32) -> torch.Tensor:
        batch_starts = torch.arange(0, len(sequences), batch_size)
        output_list = []
        for batch_start in batch_starts:
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            # Truncate/pad sequences if we mandate constant length, do not otherwise:
            batch_inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding='max_length', max_length=self.max_tokenized_length, truncation=True)
            output_list.append(self.__embed_batch_inputs(batch_inputs))

        return torch.cat(output_list, axis=0)