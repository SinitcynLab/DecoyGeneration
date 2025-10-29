import torch

from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable
from transformers import BertModel, BertTokenizer

class ProtBertEncoder(PeptideEncoder):
    def __init__(self, max_tokenized_length : int = 64):
        PeptideEncoder.__init__(self)
        self.max_tokenized_length = max_tokenized_length
        MODEL_NAME = "Rostlab/prot_bert_bfd_localization"
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

    def __call__(self, sequences: Iterable[str], batch_size : int = 32) -> torch.Tensor:
        sequences = [" ".join(sequence) for sequence in sequences]
        encodings = self.tokenizer.batch_encode_plus(sequences, 
                                   truncation=True, 
                                   add_special_tokens=True, 
                                   return_token_type_ids=False,
                                   return_attention_mask=True,
                                   max_length = self.max_tokenized_length,
                                   padding='max_length',
                                   return_tensors='pt')
        print("tokenized")
        return self.__embed_batched(encodings)

    def __embed_batched(self, encodings):
        N = encodings['input_ids'].shape[0]
        pooler_outputs = torch.zeros(N, 1024)
        for idx in range(N):
            input_ids = encodings['input_ids'][idx].unsqueeze(0)
            attention_mask = encodings['attention_mask'][idx].unsqueeze(0)
            embedding = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooler_outputs[idx] = embedding.pooler_output
            print(idx)
        return pooler_outputs
