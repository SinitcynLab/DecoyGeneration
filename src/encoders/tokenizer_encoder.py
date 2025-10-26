import torch
import numpy as np
import tensorflow as tf

from src.encoders.peptide_encoder import PeptideEncoder
from typing import Iterable
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

class TokenizerEncoder(PeptideEncoder):
    def __init__(self, seq_len : int = 512):
        PeptideEncoder.__init__(self)
        self.seq_len = seq_len
        with tf.device('CPU'):
            self.pretrained_model_generator, self.input_encoder = load_pretrained_model()
            self.model = get_model_with_hidden_layers_as_outputs(self.pretrained_model_generator.create_model(seq_len))

    def __call__(self, sequences: Iterable[str], batch_size : int = 32) -> torch.Tensor:
        for idx, sequence in enumerate(sequences):
            if len(sequence) > self.seq_len - 2:
                sequences[idx] = sequence[0:self.seq_len-2]
        with tf.device('CPU'):
            x = self.input_encoder.encode_X(sequences, self.seq_len)
        x = x[0]
        x = torch.FloatTensor(x)
        return x