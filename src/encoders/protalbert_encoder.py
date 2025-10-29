import torch
import re
from transformers import AlbertModel, AlbertTokenizer
from typing import Iterable

from src.encoders.peptide_encoder import PeptideEncoder

class ProtAlbertEncoder(PeptideEncoder):
    def __init__(self, max_tokenized_length : int = 256):
        PeptideEncoder.__init__(self)
        torch.set_printoptions(threshold=100)
        self.max_tokenized_length = max_tokenized_length
        self.tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
        self.model = AlbertModel.from_pretrained("Rostlab/prot_albert")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, sequences : Iterable[str]) -> torch.Tensor:
        sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences] # map rare residues to X
        sequences = [" ".join(sequence) for sequence in sequences] # Add white spaces between characters
        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, truncation=True, max_length=self.max_tokenized_length, padding="max_length")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        with torch.no_grad():
            embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        features = torch.zeros(len(sequences), 4096)
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features[seq_num] = torch.FloatTensor(seq_emd).amax(dim=0)
        return features
