"""Decoder-only Transformer for peptide language modeling."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, layer_norm_eps):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = PositionalEmbedding(config.d_model, config.max_len)
        self.feat_proj = None
        if getattr(config, 'features_dim', 0):
            self.feat_proj = nn.Linear(config.features_dim, config.d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout, config.layer_norm_eps)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: (B, L)
        B, L = input_ids.shape
        device = input_ids.device
        if L > self.config.max_len:
            raise ValueError("Sequence length exceeds model max_len")
        tok = self.token_emb(input_ids)
        pos = self.pos_emb(tok)
        x = tok + pos
        if features is not None and self.feat_proj is not None:
            # features: (B, L, F)
            x = x + self.feat_proj(features)
        # causal mask (L,L) with True = mask (following nn.MultiheadAttention semantics expects float mask?)
        causal_mask = torch.full((L, L), float('-inf'), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # key_padding_mask: True for pads
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # (B, L)
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits  # (B, L, V)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float, vocab_size: int, ignore_index: int):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,L,V), target: (B,L)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            ignore_mask = target == self.ignore_index
            target_clamped = target.clone()
            target_clamped[ignore_mask] = 0
            true_dist.scatter_(2, target_clamped.unsqueeze(-1), 1.0 - self.smoothing)
            true_dist[ignore_mask] = 0
        loss = -(true_dist * log_probs).sum(dim=-1)
        denom = (~ignore_mask).sum(dim=-1)
        loss = loss.sum() / denom.sum().clamp(min=1)
        return loss

@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9, repetition_penalty: float = 1.0, used_ids: Optional[torch.Tensor] = None):
    # logits: (V,)
    logits = logits / max(temperature, 1e-6)
    if used_ids is not None and repetition_penalty != 1.0:
        for uid in used_ids.tolist():
            logits[uid] /= repetition_penalty
    probs = F.softmax(logits, dim=-1)
    # nucleus filter
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    # ensure at least one token remains
    if mask[0]:
        mask[0] = False
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx[idx].item()
