import torch
from decoygen.vocab import DEFAULT_VOCAB
from decoygen.config import ModelConfig, GenerationConfig
from decoygen.model import DecoderOnlyTransformer


def test_forward_and_sample():
    vocab = DEFAULT_VOCAB
    cfg = ModelConfig(vocab_size=vocab.size, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=32)
    model = DecoderOnlyTransformer(cfg)
    seq = torch.tensor([vocab.bos_id, vocab.token_to_id['A'], vocab.token_to_id['K'], vocab.eos_id])
    batch = seq.unsqueeze(0)
    logits = model(batch)
    assert logits.shape == (1, batch.size(1), vocab.size)
