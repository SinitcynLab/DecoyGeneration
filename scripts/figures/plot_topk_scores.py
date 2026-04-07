#!/usr/bin/env python3
"""Plot rank vs score for top-k suspicious targets across experiments.

Usage
-----
python plot_topk_scores.py \\
    --experiments name1:diagnostics_dir1 name2:diagnostics_dir2 \\
    --top-k 200 \\
    --out plot.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Plot rank vs score for top-k suspicious targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiments", nargs="+", required=True,
        help="Experiments as name:diagnostics_dir pairs",
    )
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--score", default="best_decoy_cosine",
                        help="Score column to plot")
    parser.add_argument("--out", required=True, help="Output plot path")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    csv_rows = []

    for spec in args.experiments:
        parts = spec.split(":")
        if len(parts) != 2:
            print(f"Error: expected name:diagnostics_dir, got: {spec}",
                  file=sys.stderr)
            sys.exit(1)
        name, diag_dir = parts
        tsv = os.path.join(diag_dir, "tables", "top_suspicious_targets.tsv")
        df = pd.read_csv(tsv, sep="\t", low_memory=False)
        if args.score not in df.columns:
            print(f"Error: column '{args.score}' not in {tsv}", file=sys.stderr)
            sys.exit(1)
        df = df.sort_values(args.score, ascending=False).head(args.top_k).reset_index(drop=True)
        ax.plot(range(1, len(df) + 1), df[args.score].values, label=name)
        for rank, score in enumerate(df[args.score].values, 1):
            csv_rows.append({"experiment": name, "rank": rank, "score": score})

    ax.set_xlabel("Rank")
    ax.set_ylabel(args.score)
    ax.set_title(f"Top-{args.top_k} suspicious targets: rank vs {args.score}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {args.out}", file=sys.stderr)

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
