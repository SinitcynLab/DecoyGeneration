from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from graphviz import Digraph
except ModuleNotFoundError:
    Digraph = None


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_peptides_tsv(path: str, limit: int = 0) -> List[str]:
    peptides: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("peptide\t"):
            raise ValueError(f"Unexpected TSV header in {path!r}: {header[:50]!r}")
        for line in f:
            if not line.strip():
                continue
            pep = line.split("\t", 1)[0].strip()
            if pep:
                peptides.append(pep)
            if limit and len(peptides) >= limit:
                break
    return peptides


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int

    @staticmethod
    def build() -> "Vocab":
        tokens = ["<pad>", "<unk>"] + AMINO_ACIDS
        stoi = {t: i for i, t in enumerate(tokens)}
        return Vocab(stoi=stoi, itos=tokens, pad_id=stoi["<pad>"], unk_id=stoi["<unk>"])

    def encode(self, peptide: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in peptide]


class PeptideDataset(Dataset):
    def __init__(self, encoded: List[List[int]], labels: List[int]):
        self.encoded = encoded
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.encoded[idx], self.labels[idx]


def collate_pad(batch: Sequence[Tuple[List[int], int]], pad_id: int):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) else 0
    x = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, lengths, y


class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, pad_id: int, dropout: float, num_layers: int = 1, rnn_type: str = "rnn"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        # Support stacked RNN or GRU layers
        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                nonlinearity="tanh",
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        # h_n: (num_layers * num_directions, batch, hidden)
        h_fwd = h_n[0]
        h_bwd = h_n[1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        h = self.dropout(h)
        logits = self.fc(h).squeeze(-1)
        return logits


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return float((preds == y).float().mean().item())


def auc_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n = int(labels.shape[0])
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order]

    ranks = np.arange(1, n + 1, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_probs[j] == sorted_probs[i]:
            j += 1
        if j - i > 1:
            ranks[i:j] = 0.5 * ((i + 1) + j)
        i = j

    sum_pos_ranks = float(ranks[sorted_labels == 1].sum())
    return (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


@dataclass
class PairConfig:
    name: str
    target_path: str
    decoy_path: str | None


@dataclass
class PeptidePairInput:
    name: str
    target_peptides: List[str]
    decoy_peptides: List[str] | None


@dataclass
class ClassifierRunResult:
    stats_rows: List[dict[str, object]]
    names: List[str]
    mean_aucs: List[float]
    std_aucs: List[float]
    device: str
    out_stats_csv: str | None = None
    out_arch_svg: str | None = None


def read_target_decoy_pairs(path: str) -> List[PairConfig]:
    pairs: List[PairConfig] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if not fields:
                continue
            if fields[0].lower() == "name":
                continue
            if len(fields) < 2:
                continue
            name = fields[0]
            target_path = fields[1]
            decoy_path = fields[2] if len(fields) > 2 else None
            if name == "target":
                decoy_path = None
            pairs.append(PairConfig(name=name, target_path=target_path, decoy_path=decoy_path))
    return pairs


def load_pair_inputs(path: str) -> List[PeptidePairInput]:
    pair_inputs: List[PeptidePairInput] = []
    for pair in read_target_decoy_pairs(path):
        pair_inputs.append(
            PeptidePairInput(
                name=pair.name,
                target_peptides=read_peptides_tsv(pair.target_path),
                decoy_peptides=read_peptides_tsv(pair.decoy_path) if pair.decoy_path else None,
            )
        )
    return pair_inputs


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def prepare_pair_classes(pair: PeptidePairInput, limit_per_class: int = 0) -> tuple[List[str], List[str]]:
    if pair.decoy_peptides:
        targets = list(pair.target_peptides)
        decoys = list(pair.decoy_peptides)
        shared = set(targets) & set(decoys)
        if shared:
            targets = [peptide for peptide in targets if peptide not in shared]
            decoys = [peptide for peptide in decoys if peptide not in shared]
        random.shuffle(targets)
        random.shuffle(decoys)
        if limit_per_class:
            targets = targets[:limit_per_class]
            decoys = decoys[:limit_per_class]
        n = min(len(targets), len(decoys))
        if n == 0:
            raise ValueError(f"No peptides loaded for pair {pair.name!r}")
        return targets[:n], decoys[:n]

    peptides = list(pair.target_peptides)
    if len(peptides) < 2:
        raise ValueError(f"Need at least 2 peptides for target-vs-target pair {pair.name!r}")
    random.shuffle(peptides)
    n = len(peptides) // 2
    if limit_per_class:
        n = min(n, limit_per_class)
    if n == 0:
        raise ValueError(f"No peptides available after split for pair {pair.name!r}")
    return peptides[:n], peptides[n : 2 * n]


def export_architecture_svg(path: str, emb_dim: int, hidden_dim: int, num_layers: int, rnn_type: str, dropout: float) -> str:
    if Digraph is None:
        raise ModuleNotFoundError("graphviz is required for --out-arch-svg; install python package 'graphviz'.")

    out_root, out_ext = os.path.splitext(path)
    if out_ext and out_ext.lower() != ".svg":
        raise ValueError("--out-arch-svg must end with .svg (or have no extension)")
    render_path = out_root if out_ext.lower() == ".svg" else path

    g = Digraph("decoy_classifier_arch", format="svg")
    g.attr(rankdir="LR")
    g.node("in", "Input peptide\\n(token IDs)")
    g.node("emb", f"Embedding\\nemb_dim={emb_dim}")
    g.node("rnn", f"Bi{rnn_type.upper()} x{num_layers}\\nhidden_dim={hidden_dim}")
    g.node("cat", "Concat h_fwd + h_bwd\\n(2 * hidden_dim)")
    g.node("drop", f"Dropout\\np={dropout}")
    g.node("fc", "Linear\\n2 * hidden_dim -> 1")
    g.node("sig", "Sigmoid\\nP(class=target)")
    for src, dst in [("in", "emb"), ("emb", "rnn"), ("rnn", "cat"), ("cat", "drop"), ("drop", "fc"), ("fc", "sig")]:
        g.edge(src, dst)

    os.makedirs(os.path.dirname(render_path) or ".", exist_ok=True)
    return g.render(render_path, cleanup=True)


def save_auc_plot(
    result: ClassifierRunResult,
    output_path: str,
    cross_validation_fold: int,
) -> str:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting; install it or skip --out-plot")

    x = np.arange(len(result.names))
    plt.figure(figsize=(max(8, len(result.names) * 1.2), 5))
    plt.bar(x, result.mean_aucs, yerr=result.std_aucs, capsize=4)
    plt.xticks(x, result.names, rotation=45, ha="right")
    plt.ylabel("Validation AUC")
    plt.title(f"{cross_validation_fold}-Fold Cross-Validation")
    plt.ylim(0.4, 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to: {output_path}")
    return output_path


def run_decoy_classifier(
    pairs: Sequence[PeptidePairInput],
    limit_per_class: int = 0,
    seed: int = 1234,
    emb_dim: int = 64,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    num_layers: int = 2,
    rnn_type: str = "gru",
    batch_size: int = 1024,
    epochs: int = 3,
    lr: float = 1e-3,
    cross_validation_fold: int = 5,
    device: str = "auto",
    out_model: str | None = None,
    out_arch_svg: str | None = None,
    arch_only: bool = False,
    out_stats_csv: str | None = None,
) -> ClassifierRunResult:
    if cross_validation_fold < 2:
        raise ValueError("--cross-validation-fold must be >= 2")
    if arch_only and not out_arch_svg:
        raise ValueError("--arch-only requires --out-arch-svg")

    set_seed(seed)
    resolved_device = resolve_device(device)

    saved_arch_path: str | None = None
    if out_arch_svg:
        saved_arch_path = export_architecture_svg(
            path=out_arch_svg,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=dropout,
        )
        print(f"Saved architecture SVG to: {saved_arch_path}")
        if arch_only:
            return ClassifierRunResult(
                stats_rows=[],
                names=[],
                mean_aucs=[],
                std_aucs=[],
                device=str(resolved_device),
                out_arch_svg=saved_arch_path,
            )

    if not pairs:
        raise ValueError("No peptide pairs were provided.")

    print(f"Device: {resolved_device}")
    names: List[str] = []
    mean_aucs: List[float] = []
    std_aucs: List[float] = []
    stats_rows: List[dict[str, object]] = []
    run_args = {
        "limit_per_class": limit_per_class,
        "seed": seed,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "num_layers": num_layers,
        "rnn_type": rnn_type,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "cross_validation_fold": cross_validation_fold,
        "device": str(resolved_device),
    }

    for pair in pairs:
        targets, decoys = prepare_pair_classes(pair, limit_per_class=limit_per_class)
        n = min(len(targets), len(decoys))
        vocab = Vocab.build()
        x_encoded = [vocab.encode(peptide) for peptide in targets] + [vocab.encode(peptide) for peptide in decoys]
        y = [1] * n + [0] * n

        idxs = list(range(len(y)))
        random.shuffle(idxs)
        x_encoded = [x_encoded[i] for i in idxs]
        y = [y[i] for i in idxs]

        if len(y) < cross_validation_fold:
            raise ValueError(
                f"Pair {pair.name!r} has {len(y)} samples, less than --cross-validation-fold={cross_validation_fold}"
            )

        folds: List[np.ndarray] = list(np.array_split(np.arange(len(y)), cross_validation_fold))
        fold_aucs: List[float] = []

        print(f"\nPair {pair.name}: {n} class-A + {n} class-B (total {2*n})")
        for fold_idx, val_idx_np in enumerate(folds):
            set_seed(seed + fold_idx)
            val_idx = [int(i) for i in val_idx_np.tolist()]
            train_idx = [int(i) for j, idx_np in enumerate(folds) if j != fold_idx for i in idx_np.tolist()]

            x_val = [x_encoded[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]
            x_train = [x_encoded[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]

            train_ds = PeptideDataset(x_train, y_train)
            val_ds = PeptideDataset(x_val, y_val)

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=lambda b: collate_pad(b, vocab.pad_id),
                drop_last=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_pad(b, vocab.pad_id),
                drop_last=False,
            )

            model = BiRNNClassifier(
                vocab_size=len(vocab.itos),
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                pad_id=vocab.pad_id,
                dropout=dropout,
                num_layers=num_layers,
                rnn_type=rnn_type,
            ).to(resolved_device)

            if fold_idx == 0:
                total_params = sum(parameter.numel() for parameter in model.parameters())
                trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")

            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss()
            print(f"Fold {fold_idx + 1}/{cross_validation_fold} | Train/val: {len(train_ds)}/{len(val_ds)}")

            final_val_auc = float("nan")
            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                train_acc = 0.0
                n_train_batches = 0

                for x, lengths, yb in train_loader:
                    x = x.to(resolved_device)
                    lengths = lengths.to(resolved_device)
                    yb = yb.to(resolved_device)

                    opt.zero_grad(set_to_none=True)
                    logits = model(x, lengths)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()

                    train_loss += float(loss.item())
                    train_acc += accuracy_from_logits(logits.detach(), yb.detach())
                    n_train_batches += 1

                model.eval()
                val_loss = 0.0
                val_acc = 0.0
                n_val_batches = 0
                val_probs: List[torch.Tensor] = []
                val_labels: List[torch.Tensor] = []
                with torch.no_grad():
                    for x, lengths, yb in val_loader:
                        x = x.to(resolved_device)
                        lengths = lengths.to(resolved_device)
                        yb = yb.to(resolved_device)
                        logits = model(x, lengths)
                        loss = loss_fn(logits, yb)
                        val_loss += float(loss.item())
                        val_acc += accuracy_from_logits(logits, yb)
                        val_probs.append(torch.sigmoid(logits).cpu())
                        val_labels.append(yb.cpu())
                        n_val_batches += 1

                train_loss /= max(1, n_train_batches)
                train_acc /= max(1, n_train_batches)
                val_loss /= max(1, n_val_batches)
                val_acc /= max(1, n_val_batches)
                val_auc = auc_from_probs(torch.cat(val_probs).numpy(), torch.cat(val_labels).numpy())
                final_val_auc = val_auc

                print(
                    f"Fold {fold_idx + 1:02d} Epoch {epoch:02d} | "
                    f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                    f"val loss {val_loss:.4f} acc {val_acc:.3f} auc {val_auc:.3f}"
                )

            fold_aucs.append(final_val_auc)

            if out_model:
                model_path = out_model
                try:
                    model_path = model_path.format(name=pair.name, fold=fold_idx + 1)
                except Exception:
                    pass
                os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "vocab": vocab.itos,
                        "pad_id": vocab.pad_id,
                        "unk_id": vocab.unk_id,
                        "args": run_args,
                        "pair_name": pair.name,
                        "fold": fold_idx + 1,
                    },
                    model_path,
                )
                print(f"Saved model to: {model_path}")

        mean_auc = float(np.nanmean(np.array(fold_aucs, dtype=np.float64)))
        std_auc = float(np.nanstd(np.array(fold_aucs, dtype=np.float64)))
        names.append(pair.name)
        mean_aucs.append(mean_auc)
        std_aucs.append(std_auc)
        stats_rows.append(
            {
                "class_a": len(targets),
                "class_b": len(decoys),
                "auc_mean": mean_auc,
                "auc_std": std_auc,
                "experiment": pair.name,
            }
        )
        print(f"Pair {pair.name} | val AUC mean/std: {mean_auc:.3f}/{std_auc:.3f}")

    if out_stats_csv:
        os.makedirs(os.path.dirname(out_stats_csv) or ".", exist_ok=True)
        with open(out_stats_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["class_a", "class_b", "auc_mean", "auc_std", "experiment"],
            )
            writer.writeheader()
            writer.writerows(stats_rows)
        print(f"Saved stats CSV to: {out_stats_csv}")

    return ClassifierRunResult(
        stats_rows=stats_rows,
        names=names,
        mean_aucs=mean_aucs,
        std_aucs=std_aucs,
        device=str(resolved_device),
        out_stats_csv=out_stats_csv,
        out_arch_svg=saved_arch_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="target_decoy_pairs.txt")
    ap.add_argument("--limit-per-class", type=int, default=0, help="Optional extra cap at load time.")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num-layers", type=int, default=2, help="Number of stacked RNN/GRU layers")
    ap.add_argument("--rnn-type", choices=["rnn", "gru"], default="gru", help="Recurrent layer type")

    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cross-validation-fold", type=int, default=5)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--out-model", default=None, help="Optional path to save model checkpoint per fold")
    ap.add_argument("--out-plot", default="decoy_classifier_cv_auc.pdf")
    ap.add_argument("--out-arch-svg", default="", help="Optional path to save a model architecture SVG")
    ap.add_argument("--arch-only", action="store_true", help="Export architecture SVG and exit without loading data or training")
    ap.add_argument(
        "--out-stats-csv",
        default="decoy_classifier_cv_stats.csv",
        help="Path to save per-experiment summary CSV; pass '' to disable",
    )

    args = ap.parse_args()
    pair_inputs = load_pair_inputs(args.pairs) if not args.arch_only else []
    result = run_decoy_classifier(
        pairs=pair_inputs,
        limit_per_class=args.limit_per_class,
        seed=args.seed,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        cross_validation_fold=args.cross_validation_fold,
        device=args.device,
        out_model=args.out_model,
        out_arch_svg=args.out_arch_svg or None,
        arch_only=args.arch_only,
        out_stats_csv=args.out_stats_csv or None,
    )
    if args.out_plot and not args.arch_only:
        save_auc_plot(result, args.out_plot, args.cross_validation_fold)


if __name__ == "__main__":
    main()
