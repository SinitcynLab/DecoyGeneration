#!/usr/bin/env python3
"""Analyze decoy similarity patterns from diagnostics outputs.

Given diagnostics results and corresponding FASTA files, computes:
1. Classification of top suspicious targets into I<->L, XY<->YX, both, rest
2. Same-index decoy matching rate (is the nearest decoy the one generated
   for this specific target peptide?)
3. Per-length breakdowns of both
4. Multi-experiment comparison bar plots

Usage
-----
python analyze_decoy_similarity.py \\
    --experiments name1:diagnostics_dir1:fasta1 name2:diagnostics_dir2:fasta2 \\
    --top-k 100 \\
    --outdir output/
"""
from __future__ import annotations

import argparse
import gzip
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def _plot_ext() -> str:
    return ".pdf" if os.environ.get("PLOT_PDFS") == "1" else ".png"


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", file=sys.stderr, flush=True)


# ── FASTA and digestion ──────────────────────────────────────────────────

def read_fasta(path: str):
    """Yield (header, sequence) pairs from a FASTA file."""
    opener = gzip.open if path.endswith(".gz") else open
    header = None
    seq_parts: List[str] = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)


def digest_trypsin(
    sequence: str,
    missed_cleavages: int,
    min_length: int,
    max_length: int,
    ignore_proline: bool = False,
) -> List[str]:
    """Tryptic digest: cleave after K/R (optionally ignoring the P rule)."""
    cut_sites = [0]
    n = len(sequence)
    for i in range(n - 1):
        if sequence[i] in ("K", "R"):
            if ignore_proline or sequence[i + 1] != "P":
                cut_sites.append(i + 1)
    cut_sites.append(n)
    peptides: List[str] = []
    for start_idx in range(len(cut_sites) - 1):
        for mc in range(missed_cleavages + 1):
            stop_idx = start_idx + mc + 1
            if stop_idx >= len(cut_sites):
                continue
            pep = sequence[cut_sites[start_idx]:cut_sites[stop_idx]]
            if min_length <= len(pep) <= max_length:
                peptides.append(pep)
    return peptides


def build_protein_dicts(
    fasta_path: str,
    decoy_tag: str = "rev_",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Parse FASTA into target and decoy protein dicts: canonical_name -> sequence."""
    targets: Dict[str, str] = {}
    decoys: Dict[str, str] = {}
    for header, seq in read_fasta(fasta_path):
        token = header.strip().split()[0]
        if token.startswith(decoy_tag):
            canonical = token[len(decoy_tag):]
            decoys[canonical] = seq
        else:
            targets[token] = seq
    return targets, decoys


# ── Classification logic ─────────────────────────────────────────────────

# Data labels (used in DataFrames / TSVs) vs display labels (used in plots).
CAT_IL = "I<->L"
CAT_SWAP = "XY<->YX"
CAT_BOTH = "I<->L, XY<->YX"
CAT_REST = "rest"

CATEGORY_ORDER = [CAT_IL, CAT_SWAP, CAT_BOTH, CAT_REST]

CATEGORY_DISPLAY = {
    CAT_IL: r"I$\leftrightarrow$L",
    CAT_SWAP: r"XY$\leftrightarrow$YX",
    CAT_BOTH: r"I$\leftrightarrow$L, XY$\leftrightarrow$YX",
    CAT_REST: "rest",
}

CATEGORY_COLORS = {
    CAT_IL: "#4e79a7",
    CAT_SWAP: "#f28e2b",
    CAT_BOTH: "#e15759",
    CAT_REST: "#76b7b2",
}


def normalize_il(seq: str) -> str:
    return seq.replace("I", "L")


def is_adjacent_swap(s1: str, s2: str) -> bool:
    """Check if s1 and s2 differ by exactly one adjacent transposition."""
    if len(s1) != len(s2):
        return False
    diffs = [i for i in range(len(s1)) if s1[i] != s2[i]]
    if len(diffs) != 2:
        return False
    i, j = diffs
    if j != i + 1:
        return False
    return s1[i] == s2[j] and s1[j] == s2[i]


def classify_pair(target_seq: str, decoy_seq: str) -> str:
    """Classify a target-decoy pair.

    Categories (checked in order):
      I<->L          : identical after I/L normalisation
      XY<->YX        : single adjacent swap (raw sequences)
      I<->L, XY<->YX : single adjacent swap after I/L normalisation
      rest           : anything else
    """
    if not target_seq or not decoy_seq or decoy_seq == "nan":
        return CAT_REST
    if len(target_seq) != len(decoy_seq):
        return CAT_REST
    if target_seq == decoy_seq:
        return CAT_REST

    norm_t = normalize_il(target_seq)
    norm_d = normalize_il(decoy_seq)

    if norm_t == norm_d:
        return CAT_IL

    if is_adjacent_swap(target_seq, decoy_seq):
        return CAT_SWAP

    if is_adjacent_swap(norm_t, norm_d):
        return CAT_BOTH

    return CAT_REST


# ── Same-index decoy matching ────────────────────────────────────────────

def find_same_index_decoys(
    query_seq: str,
    protein_roots: List[str],
    target_proteins: Dict[str, str],
    decoy_proteins: Dict[str, str],
    missed_cleavages: int,
    min_length: int,
    max_length: int,
    ignore_proline: bool = False,
) -> List[str]:
    """Return all possible same-index decoy peptides for a target query."""
    results: List[str] = []
    for root in protein_roots:
        if root not in target_proteins or root not in decoy_proteins:
            continue
        target_peps = digest_trypsin(
            target_proteins[root], missed_cleavages, min_length, max_length,
            ignore_proline=ignore_proline,
        )
        decoy_peps = digest_trypsin(
            decoy_proteins[root], missed_cleavages, min_length, max_length,
            ignore_proline=ignore_proline,
        )
        for idx, pep in enumerate(target_peps):
            if pep == query_seq and idx < len(decoy_peps):
                results.append(decoy_peps[idx])
    return results


# ── Experiment processing ────────────────────────────────────────────────

def process_classification(
    name: str,
    diagnostics_dir: str,
    top_k: int,
) -> pd.DataFrame:
    tsv_path = os.path.join(diagnostics_dir, "tables", "top_suspicious_targets.tsv")
    df = pd.read_csv(tsv_path, sep="\t", nrows=top_k)
    results = []
    for _, row in df.iterrows():
        target = str(row["query_sequence"])
        decoy = str(row["best_decoy_sequence_cosine"])
        results.append({
            "experiment": name,
            "query_sequence": target,
            "query_length": int(row["query_length"]),
            "nearest_decoy": decoy,
            "category": classify_pair(target, decoy),
        })
    return pd.DataFrame(results)


def process_same_index(
    name: str,
    diagnostics_dir: str,
    fasta_path: str,
    missed_cleavages: int,
    min_length: int,
    max_length: int,
    ignore_proline: bool = False,
) -> pd.DataFrame:
    tsv_path = os.path.join(diagnostics_dir, "tables", "nonstealability.tsv")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    target_proteins, decoy_proteins = build_protein_dicts(fasta_path)
    log(f"  Loaded {len(target_proteins)} target and {len(decoy_proteins)} decoy proteins")

    results = []
    for _, row in df.iterrows():
        query = str(row["query_sequence"])
        nearest = str(row["best_decoy_sequence_cosine"])
        same_protein = (
            str(row.get("best_decoy_same_protein_cross_version_cosine_category", ""))
            == "same_protein_cross_version"
        )
        roots = [r for r in str(row["query_protein_roots"]).split(";") if r]

        expected_list = find_same_index_decoys(
            query, roots, target_proteins, decoy_proteins,
            missed_cleavages, min_length, max_length,
            ignore_proline=ignore_proline,
        )

        if not expected_list:
            match = "unresolved"
        elif same_protein and nearest in expected_list:
            match = "same"
        else:
            match = "different"

        results.append({
            "experiment": name,
            "query_sequence": query,
            "query_length": int(row["query_length"]),
            "nearest_decoy": nearest,
            "expected_decoys": ";".join(expected_list),
            "match": match,
        })
    return pd.DataFrame(results)


# ── Plotting ─────────────────────────────────────────────────────────────

MATCH_ORDER = ["same", "different"]
MATCH_COLORS = {"same": "#59a14f", "different": "#e15759", "unresolved": "#bab0ac"}


LENGTH_BINS = list(range(7, 16)) + [16]  # 7..15 individual, 16 = "16+"
LENGTH_LABELS = {i: str(i) for i in range(7, 16)}
LENGTH_LABELS[16] = "16+"


def _bucket_length(length: int) -> int:
    """Map peptide length to a plot bucket (16 means 16+)."""
    return length if length <= 15 else 16


def _stacked_bar_overall(
    ax,
    categories: List[str],
    experiments: List[str],
    counts: pd.DataFrame,
    colors: Dict[str, str],
    display_labels: Optional[Dict[str, str]] = None,
):
    """One stacked bar per experiment."""
    x = np.arange(len(experiments))
    bottom = np.zeros(len(experiments))
    for cat in categories:
        vals = counts[cat].reindex(experiments, fill_value=0).values.astype(float)
        label = display_labels[cat] if display_labels else cat
        ax.bar(x, vals, bottom=bottom, label=label, color=colors[cat])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")


def _stacked_bar_by_length(
    ax,
    sub: pd.DataFrame,
    group_col: str,
    order: List[str],
    colors: Dict[str, str],
    display_labels: Optional[Dict[str, str]] = None,
):
    """Stacked bar chart with bucketed lengths on x (7..15, 16+)."""
    tmp = sub.copy()
    tmp["_len_bucket"] = tmp["query_length"].apply(_bucket_length)
    ct = tmp.groupby(["_len_bucket", group_col]).size().unstack(fill_value=0)
    ct = ct.reindex(LENGTH_BINS, fill_value=0)
    for cat in order:
        if cat not in ct.columns:
            ct[cat] = 0
    ct = ct[order]
    x = np.arange(len(LENGTH_BINS))
    bottom = np.zeros(len(x))
    for cat in order:
        vals = ct[cat].values.astype(float)
        label = display_labels[cat] if display_labels else cat
        ax.bar(x, vals, bottom=bottom, label=label, color=colors[cat])
        bottom += vals
    ax.set_xlabel("Peptide length")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels([LENGTH_LABELS[b] for b in LENGTH_BINS])
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)


def plot_classification_overall(all_df: pd.DataFrame, outdir: pathlib.Path):
    ext = _plot_ext()
    experiments = list(all_df["experiment"].unique())
    counts = all_df.groupby(["experiment", "category"]).size().unstack(fill_value=0)
    for cat in CATEGORY_ORDER:
        if cat not in counts.columns:
            counts[cat] = 0

    fig, ax = plt.subplots(figsize=(max(6, len(experiments) * 1.2 + 2), 5))
    _stacked_bar_overall(ax, CATEGORY_ORDER, experiments, counts, CATEGORY_COLORS,
                         display_labels=CATEGORY_DISPLAY)
    ax.set_ylabel("Count")
    ax.set_title("Top suspicious targets: nearest decoy classification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"classification_overall{ext}", dpi=150)
    plt.close(fig)


def plot_classification_per_length(all_df: pd.DataFrame, outdir: pathlib.Path):
    ext = _plot_ext()
    experiments = list(all_df["experiment"].unique())
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 5 * n), 5), sharey=True, squeeze=False)
    for idx, exp in enumerate(experiments):
        ax = axes[0, idx]
        sub = all_df[all_df["experiment"] == exp]
        _stacked_bar_by_length(ax, sub, "category", CATEGORY_ORDER, CATEGORY_COLORS,
                               display_labels=CATEGORY_DISPLAY)
        ax.set_title(exp)
        if idx == 0:
            ax.legend(fontsize="small")
    fig.suptitle("Nearest decoy classification by peptide length", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / f"classification_per_length{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_same_index_overall(all_df: pd.DataFrame, outdir: pathlib.Path):
    ext = _plot_ext()
    df = all_df[all_df["match"] != "unresolved"]
    experiments = list(all_df["experiment"].unique())
    counts = df.groupby(["experiment", "match"]).size().unstack(fill_value=0)
    for m in MATCH_ORDER:
        if m not in counts.columns:
            counts[m] = 0
    totals = counts.sum(axis=1).replace(0, 1)
    fracs = counts.div(totals, axis=0)

    fig, ax = plt.subplots(figsize=(max(6, len(experiments) * 1.2 + 2), 5))
    _stacked_bar_overall(ax, MATCH_ORDER, experiments, fracs, MATCH_COLORS)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Nearest decoy: same-index match rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"same_index_overall{ext}", dpi=150)
    plt.close(fig)


def _stacked_bar_by_length_relative(
    ax,
    sub: pd.DataFrame,
    group_col: str,
    order: List[str],
    colors: Dict[str, str],
    display_labels: Optional[Dict[str, str]] = None,
):
    """Stacked bar chart with bucketed lengths, normalized to fraction (0–1)."""
    tmp = sub.copy()
    tmp["_len_bucket"] = tmp["query_length"].apply(_bucket_length)
    ct = tmp.groupby(["_len_bucket", group_col]).size().unstack(fill_value=0)
    ct = ct.reindex(LENGTH_BINS, fill_value=0)
    for cat in order:
        if cat not in ct.columns:
            ct[cat] = 0
    ct = ct[order]
    totals = ct.sum(axis=1).replace(0, 1)
    ct = ct.div(totals, axis=0)
    x = np.arange(len(LENGTH_BINS))
    bottom = np.zeros(len(x))
    for cat in order:
        vals = ct[cat].values.astype(float)
        label = display_labels[cat] if display_labels else cat
        ax.bar(x, vals, bottom=bottom, label=label, color=colors[cat])
        bottom += vals
    ax.set_xlabel("Peptide length")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([LENGTH_LABELS[b] for b in LENGTH_BINS])
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)


def plot_same_index_per_length(all_df: pd.DataFrame, outdir: pathlib.Path):
    ext = _plot_ext()
    df = all_df[all_df["match"] != "unresolved"]
    experiments = list(all_df["experiment"].unique())
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 5 * n), 5), sharey=True, squeeze=False)
    for idx, exp in enumerate(experiments):
        ax = axes[0, idx]
        sub = df[df["experiment"] == exp]
        _stacked_bar_by_length_relative(ax, sub, "match", MATCH_ORDER, MATCH_COLORS)
        ax.set_title(exp)
        if idx == 0:
            ax.legend(fontsize="small")
    fig.suptitle("Same-index decoy match by peptide length", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / f"same_index_per_length{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze decoy similarity patterns from diagnostics outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiments", nargs="+", required=True,
        help="Experiments as name:diagnostics_dir:fasta_path triples",
    )
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of top suspicious targets to classify")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--missed-cleavages", type=int, default=2)
    parser.add_argument("--min-length", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--decoy-tag", default="rev_")
    parser.add_argument("--ignore-proline", action="store_true",
                        help="Ignore the proline rule (KP/RP) in tryptic digestion "
                             "for same-index matching")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tables_dir = outdir / "tables"
    tables_dir.mkdir(exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Parse experiments
    experiments = []
    for spec in args.experiments:
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"Error: expected name:diagnostics_dir:fasta_path, got: {spec}",
                  file=sys.stderr)
            sys.exit(1)
        experiments.append({"name": parts[0], "diag_dir": parts[1], "fasta": parts[2]})

    # ── Analysis 1: classification ────────────────────────────────────────
    all_class: List[pd.DataFrame] = []
    for exp in experiments:
        log(f"Classifying top-{args.top_k} for {exp['name']} ...")
        df = process_classification(exp["name"], exp["diag_dir"], args.top_k)
        all_class.append(df)
    all_class_df = pd.concat(all_class, ignore_index=True)

    # ── Analysis 2: same-index matching ───────────────────────────────────
    all_match: List[pd.DataFrame] = []
    for exp in experiments:
        log(f"Same-index matching for {exp['name']} ...")
        df = process_same_index(
            exp["name"], exp["diag_dir"], exp["fasta"],
            args.missed_cleavages, args.min_length, args.max_length,
            ignore_proline=args.ignore_proline,
        )
        all_match.append(df)
    all_match_df = pd.concat(all_match, ignore_index=True)

    # ── Save tables ───────────────────────────────────────────────────────
    all_class_df.to_csv(tables_dir / "classification.tsv", sep="\t", index=False)
    all_match_df.to_csv(tables_dir / "same_index_matching.tsv", sep="\t", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_classification_overall(all_class_df, plots_dir)
    plot_classification_per_length(all_class_df, plots_dir)
    plot_same_index_overall(all_match_df, plots_dir)
    plot_same_index_per_length(all_match_df, plots_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    log("Classification summary:")
    print(all_class_df.groupby(["experiment", "category"]).size()
          .unstack(fill_value=0).to_string(), file=sys.stderr)
    log("Same-index matching summary:")
    print(all_match_df.groupby(["experiment", "match"]).size()
          .unstack(fill_value=0).to_string(), file=sys.stderr)
    log(f"Done. Output in {outdir}")


if __name__ == "__main__":
    main()
