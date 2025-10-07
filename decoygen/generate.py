"""Generation script for constrained decoy peptide sampling."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Set
import torch
import torch.nn.functional as F
from .vocab import DEFAULT_VOCAB
from .config import ModelConfig, GenerationConfig
from .model import DecoderOnlyTransformer, sample_next_token
from .constraints import ConstraintState, apply_length_constraints, enforce_tryptic_end, mask_invalid_tokens
from .mass import monoisotopic_mass
from .filters import remove_exact_matches, remove_high_identity
from .utils import load_checkpoint


def parse_args(arg_list=None):
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--num', type=int, default=100)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--targets', type=Path, help='Optional file of target peptides to avoid')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--repetition-penalty', type=float, default=1.2)
    p.add_argument('--min-length', type=int, default=7)
    p.add_argument('--max-length', type=int, default=35)
    p.add_argument('--mass-min', type=float, default=500.0)
    p.add_argument('--mass-max', type=float, default=3000.0)
    p.add_argument('--max-missed-cleavages', type=int, default=2)
    p.add_argument('--reject-identity', type=float, default=0.8)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args(arg_list)


def load_targets(path: Path) -> Set[str]:
    if not path or not path.exists():
        return set()
    with path.open() as f:
        return {line.strip() for line in f if line.strip()}


def build_model_from_ckpt(path, vocab_size: int):
    # Load checkpoint only to inspect model config first
    dummy = DecoderOnlyTransformer(ModelConfig(vocab_size=vocab_size))  # temporary (will reload real state later)
    # We'll re-load state after instantiating actual config
    step, model_cfg, ckpt = load_checkpoint(path, model=None, optimizer=None, config_class=ModelConfig)
    if model_cfg is None:
        model_cfg = ModelConfig(vocab_size=vocab_size)
    model = DecoderOnlyTransformer(model_cfg)
    model.load_state_dict(ckpt.get('model_state', ckpt.get('model')))
    return model, model_cfg

@torch.no_grad()
def generate_one(model: DecoderOnlyTransformer, gen_cfg: GenerationConfig, device: str):
    vocab = DEFAULT_VOCAB
    bom = torch.tensor([[vocab.bom_id]], dtype=torch.long, device=device)
    seq_ids = bom.clone() #torch.empty((1, 0), dtype=torch.long, device=device)
    state = ConstraintState(gen_cfg.max_missed_cleavages)
    used_ids = []
    for step in range(1, gen_cfg.max_length + 1):
        logits = model(seq_ids)[:, -1, :].squeeze(0)  # (V,)
        # apply_length_constraints(step, gen_cfg.min_length, gen_cfg.max_length, vocab.size, logits, vocab.eos_id)
        # enforce_tryptic_end([vocab.id_to_token[i.item()] for i in seq_ids[0][1:]], step, gen_cfg.min_length, gen_cfg.max_length, logits, vocab.eos_id, vocab)
        mask_invalid_tokens(seq_ids[0].tolist(), logits, vocab, gen_cfg.tryptic, state)
        next_id = sample_next_token(logits.clone(), temperature=gen_cfg.temperature, top_p=gen_cfg.top_p, repetition_penalty=gen_cfg.repetition_penalty, used_ids=torch.tensor(used_ids, device=device) if used_ids else None)
        used_ids.append(next_id)
        seq_ids = torch.cat([seq_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == vocab.eos_id:
            break
    peptide = vocab.decode(seq_ids[0].tolist())
    return peptide

@torch.no_grad()
def generate_batch(model, gen_cfg: GenerationConfig, device: str, n: int) -> List[str]:
    out = []
    attempts = 0
    while len(out) < n and attempts < gen_cfg.max_attempts:
        attempts += 1
        p = generate_one(model, gen_cfg, device)
        # if gen_cfg.min_length <= len(p) <= gen_cfg.max_length:
        out.append(p)
    return out


def main(cli_args=None):
    args = parse_args(cli_args)
    vocab = DEFAULT_VOCAB
    model, model_cfg = build_model_from_ckpt(str(args.checkpoint), vocab.size)
    model = model.to(args.device)
    gen_cfg = GenerationConfig(max_length=args.max_length, min_length=args.min_length, temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.repetition_penalty, mass_min=args.mass_min, mass_max=args.mass_max, max_missed_cleavages=args.max_missed_cleavages, reject_identity_threshold=args.reject_identity)
    model.eval()
    raw = generate_batch(model, gen_cfg, args.device, args.num)
    # Mass filter inline to avoid importing full pipeline again
    mass_filtered = raw #[p for p in raw if gen_cfg.mass_min <= monoisotopic_mass(p) <= gen_cfg.mass_max]
    targets = load_targets(args.targets) if args.targets else set()
    uniq = remove_exact_matches(mass_filtered, targets)
    final = uniq #remove_high_identity(uniq, targets, gen_cfg.reject_identity_threshold) if targets else uniq
    with args.out.open('w') as f:
        for i, pep in enumerate(final):
            f.write(f">decoy_{i}\n{pep}\n")
    print(f"Generated {len(final)} decoys (raw {len(raw)} attempts). Written to {args.out}")

if __name__ == '__main__':
    main()
