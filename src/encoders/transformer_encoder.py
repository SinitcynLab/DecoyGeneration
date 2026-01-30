import torch
from typing import Iterable, Tuple, List

from src.encoders.peptide_encoder import PeptideEncoder

class TransformerEncoder(PeptideEncoder):
    def __init__(self, max_tokenized_length : int = 256, device : torch.device = 'cpu', constant_length : bool = True, flatten: bool = True, 
                 cls_only : bool = False):
        if flatten and not constant_length:
            raise ValueError("Varied-length encodings are incompatible with flattened encodings.")
        PeptideEncoder.__init__(self)
        self.max_tokenized_length = max_tokenized_length
        self.device = device
        self.constant_length = constant_length
        self.flatten=flatten
        self.cls_only = cls_only
    
    def __extract_hidden_state(self, output_object) -> torch.Tensor:
        # Extract hidden state in one of two ways (different for Protbert and ESM):
        if hasattr(output_object, 'last_hidden_state'):
            hidden_states = output_object.last_hidden_state # extract the embeddings
        else:
            hidden_states = output_object.hidden_states[-1] # extract the embeddings
        
        # Return only CLS from hidden state if desired:
        if self.cls_only:
            return self.__extract_cls_token(hidden_states)
        else:
            return hidden_states
        
    def __extract_cls_token(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states[:,0,:] # [batch_size, num_tokens, 1024 or 320], first token is CLS token
        return hidden_states
    
    def __embed_batch_inputs(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        batch_inputs = batch_inputs.to(self.device)
        with torch.no_grad():
            batch_outputs = self.model(**batch_inputs, output_hidden_states=True)
        batch_hidden_st_gpu = self.__extract_hidden_state(batch_outputs)
        batch_hidden_st_cpu = batch_hidden_st_gpu.cpu()
        del batch_inputs, batch_hidden_st_gpu
        torch.cuda.empty_cache()
        return batch_hidden_st_cpu

    def _embed_batched_varied_length(self, sequences : Iterable[str]) -> Iterable[torch.Tensor]:
        output_list = []
        for i in range(len(sequences)):
            batch_sequences = sequences[i:i+1]
            batch_inputs = self.tokenizer(batch_sequences, return_tensors="pt")
            output_list.append(self.__embed_batch_inputs(batch_inputs))
        
        # return output as a list of varied-length tensors:
        return output_list

    def _embed_batched_constant_length(self, sequences : Iterable[str], batch_size : int = 32) -> torch.Tensor:
        batch_starts = torch.arange(0, len(sequences), batch_size)
        output_list = []
        for batch_start in batch_starts:
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            # Truncate/pad sequences if we mandate constant length, do not otherwise:
            batch_inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding='max_length', 
                                          max_length=self.max_tokenized_length, truncation=True)
            output_list.append(self.__embed_batch_inputs(batch_inputs))

        # return output as a list of same-length tensors:
        return output_list
    
    def __call__(self, sequences : Iterable[str]) -> List[torch.Tensor]:
        if self.constant_length:
            embeddings: torch.Tensor = self._embed_batched_constant_length(sequences, batch_size = 1) # [Batch, max_tokenized_length, 1024 or 320]
        else:
            embeddings: torch.Tensor = self._embed_batched_varied_length(sequences) # List, each entry [tokenized_length, 1024 or 320]
        
        if self.flatten and self.constant_length:
            embeddings: List[torch.Tensor] = [e.flatten(start_dim=1, end_dim=2) for e in embeddings] # [Batch, max_tokenized_length * (1024 or 320)]
        
        return embeddings

def pad_tensor_list(tensor_list: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Converts a given list of tensors to a single tensor, padded with zeroes where needed
    lengths: List[int] = [t.size(dim=1) for t in tensor_list] # list lengths of tensors
    lengths = torch.IntTensor(lengths)
    max_len: int = torch.max(lengths) # find the maximum length
    dim = tensor_list[0].size(dim=2) # find the encoding size (1024 or 320)
    for i in range(len(tensor_list)):
        diff = max_len - tensor_list[i].size(dim=1) # how long the padding should be in the sequence-length dimension
        pad = torch.zeros((1, diff, dim)) # create the padding as a torch tensor
        tensor_list[i] = torch.cat((tensor_list[i], pad), axis=1) # concat the tensor and padding along sequence-length dimension
    return torch.cat(tensor_list, axis=0), lengths # return tensor