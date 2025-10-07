"""Configuration dataclasses for model, training, and generation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 64
    layer_norm_eps: float = 1e-5
    features_dim: int = 0  # optional auxiliary feature dimension concatenated per token

@dataclass
class TrainingConfig:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    total_steps: int = 20000
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 100
    save_interval: int = 1000

@dataclass
class GenerationConfig:
    max_length: int = 35
    min_length: int = 7
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    num_return_sequences: int = 1
    mass_min: float = 500.0
    mass_max: float = 3000.0
    max_missed_cleavages: int = 2
    tryptic: bool = True
    reject_identity_threshold: float = 0.8  # 80% identity vs targets
    max_attempts: int = 1000

