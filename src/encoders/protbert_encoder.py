import torch
import numpy as np

from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable
from transformers import AutoTokenizer, AutoModel, pipeline

class ProtBertEncoder(PeptideEncoder):
    def __init__(self):
        PeptideEncoder.__init__(self)
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert")

    def __call__(self, sequences: Iterable[str], batch_size = 10) -> torch.Tensor:
        x = torch.zeros((len(sequences), 1024))
        batch_starts = np.arange(0, len(sequences), batch_size)
        for batch_start in batch_starts:
            batch_end = min(batch_start + batch_size, len(sequences) - 1)
            tokens = self.tokenizer(sequences[batch_start:batch_end], return_tensors='pt')
            model_output = self.model(**tokens)
            x[batch_start:batch_end, :] = model_output.last_hidden_state.mean(dim = 1)
            print("%d / %d \n" % (batch_end + 1, len(sequences)))
        print(x.shape)
        #x = self.normalize_tensor(x)
        return x