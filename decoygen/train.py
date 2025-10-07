"""Training script for decoder-only peptide LM."""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .vocab import DEFAULT_VOCAB
from .data import PeptideDataset, collate_batch
from .model import DecoderOnlyTransformer, LabelSmoothingCrossEntropy
from .config import ModelConfig, TrainingConfig
from .utils import set_seed, cosine_warmup_lr, save_checkpoint


def parse_args(arg_list=None):
    p = argparse.ArgumentParser(description="Train decoder-only Transformer on peptide sequences")
    p.add_argument('--data', type=Path, required=True, help='Text file with one peptide per line')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--ff', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(arg_list)


def load_peptides(path: Path):
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def main(cli_args=None):
    args = parse_args(cli_args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    peptides = load_peptides(args.data)
    vocab = DEFAULT_VOCAB
    # No auxiliary features; token-only modeling
    model_cfg = ModelConfig(vocab_size=vocab.size, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, d_ff=args.ff, features_dim=0)
    train_cfg = TrainingConfig(batch_size=args.batch_size, lr=args.lr, warmup_steps=args.warmup, label_smoothing=args.label_smoothing, total_steps=args.steps, device=args.device)

    set_seed(train_cfg.seed)
    dataset = PeptideDataset(peptides, vocab=vocab, max_len=model_cfg.max_len)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, vocab.pad_id))

    model = DecoderOnlyTransformer(model_cfg).to(train_cfg.device)
    criterion = LabelSmoothingCrossEntropy(train_cfg.label_smoothing, vocab_size=vocab.size, ignore_index=vocab.pad_id)
    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    step = 0
    model.train()
    while step < train_cfg.total_steps:
        for batch in loader:
            step += 1
            input_ids = batch['input_ids'].to(train_cfg.device)
            attention_mask = batch['attention_mask'].to(train_cfg.device)
            labels = batch['labels'].to(train_cfg.device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
            lr = cosine_warmup_lr(step, train_cfg.lr, train_cfg.warmup_steps, train_cfg.total_steps)
            for g in optimizer.param_groups:
                g['lr'] = lr
            optimizer.step()
            if step % train_cfg.log_interval == 0:
                print(f"step={step} loss={loss.item():.4f} lr={lr:.3e}")
            if step % train_cfg.save_interval == 0:
                save_checkpoint(str(args.out_dir / f"step_{step}.pt"), model, optimizer, step, model_config=model_cfg)
            if step >= train_cfg.total_steps:
                break
    save_checkpoint(str(args.out_dir / "final.pt"), model, optimizer, step, model_config=model_cfg)

if __name__ == '__main__':
    main()
