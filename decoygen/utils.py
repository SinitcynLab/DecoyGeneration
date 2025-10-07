"""Utility helpers."""
from __future__ import annotations
import math
import random
from dataclasses import asdict, is_dataclass
import torch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path: str, model, optimizer, step: int, model_config=None, extra: dict | None = None):
    payload = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer else None,
        'step': step,
        'version': 1,
    }
    if model_config is not None:
        if is_dataclass(model_config):
            payload['model_config'] = asdict(model_config)
        else:
            payload['model_config'] = model_config
    if extra:
        payload['extra'] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, model=None, optimizer=None, config_class=None):
    ckpt = torch.load(path, map_location='cpu')
    if model is not None:
        state_key = 'model_state' if 'model_state' in ckpt else 'model'
        model.load_state_dict(ckpt[state_key])
    if optimizer is not None and ('optimizer_state' in ckpt or 'optimizer' in ckpt):
        opt_key = 'optimizer_state' if 'optimizer_state' in ckpt else 'optimizer'
        if ckpt[opt_key] is not None:
            optimizer.load_state_dict(ckpt[opt_key])
    model_config = None
    if 'model_config' in ckpt and config_class is not None:
        model_config = config_class(**ckpt['model_config'])
    return ckpt.get('step', 0), model_config, ckpt
