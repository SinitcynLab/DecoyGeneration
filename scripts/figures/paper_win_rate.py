#!/usr/bin/env python3
"""
Merge target-win-rate curves from multiple experiments into paper-quality plots.

Produces four plots:
  1. Target-win rate by length (target queries)
  2. Target-win rate by length (decoy queries)
  3. Target-win rate by mass (random null queries, target queries)
  4. Target-win rate by mass (random null queries, decoy queries)

Usage:
    python paper_win_rate.py \\
        /path/to/tables1 "Experiment A" \\
        /path/to/tables2 "Experiment B" \\
        -o output_prefix

Each positional argument pair is: tables_dir label.

Outputs (for each plot):
    {output_prefix}_{kind}.pdf   – the figure
    {output_prefix}_{kind}.csv   – raw data sufficient to reproduce the plot
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wilson_interval(successes: float, total: float, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))
    return max(0.0, center - half), min(1.0, center + half)


def compute_win_rate_by_length(
    df: pd.DataFrame,
    winner_col: str,
    min_count: int,
) -> pd.DataFrame:
    rows = []
    for length, grp in df.groupby("query_length"):
        n = len(grp)
        if n < min_count:
            continue
        vals = grp[winner_col].astype(str)
        valid = vals[vals != "missing"]
        if len(valid) == 0:
            continue
        eff = float((valid == "target").sum()) + 0.5 * float((valid == "tie").sum())
        rate = eff / len(valid)
        lo, hi = wilson_interval(eff, len(valid))
        rows.append({"query_length": int(length), "rate": rate, "lo": lo, "hi": hi, "n": len(valid)})
    return pd.DataFrame(rows).sort_values("query_length") if rows else pd.DataFrame()


def compute_win_rate_by_mass(
    df: pd.DataFrame,
    winner_col: str,
    mass_col: str,
    min_count: int,
    n_bins: int,
) -> pd.DataFrame:
    temp = df[[mass_col, winner_col]].copy()
    temp = temp[np.isfinite(temp[mass_col])]
    if temp.empty:
        return pd.DataFrame()
    edges = np.unique(np.quantile(temp[mass_col], np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return pd.DataFrame()
    temp["mass_bin"] = pd.cut(temp[mass_col], bins=edges, include_lowest=True, duplicates="drop")
    rows = []
    for _, grp in temp.groupby("mass_bin", observed=False):
        n = len(grp)
        if n < min_count:
            continue
        vals = grp[winner_col].astype(str)
        valid = vals[vals != "missing"]
        if len(valid) == 0:
            continue
        eff = float((valid == "target").sum()) + 0.5 * float((valid == "tie").sum())
        rate = eff / len(valid)
        lo, hi = wilson_interval(eff, len(valid))
        rows.append({
            "mass_center": float(np.nanmedian(grp[mass_col])),
            "rate": rate, "lo": lo, "hi": hi, "n": len(valid),
        })
    return pd.DataFrame(rows).sort_values("mass_center") if rows else pd.DataFrame()


def plot_multi_experiment(
    results: List[Tuple[str, pd.DataFrame]],
    x_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    all_csv = []

    for label, rate_df in results:
        if rate_df.empty:
            continue
        x = rate_df[x_col].to_numpy(dtype=float)
        y = rate_df["rate"].to_numpy(dtype=float)
        lo = rate_df["lo"].to_numpy(dtype=float)
        hi = rate_df["hi"].to_numpy(dtype=float)
        line, = ax.plot(x, y, label=label, marker="o", markersize=3)
        color = line.get_color()
        ax.fill_between(x, lo, hi, alpha=0.15, color=color)

        csv_part = rate_df.copy()
        csv_part["experiment"] = label
        all_csv.append(csv_part)

    ax.axhline(0.5, linestyle="--", linewidth=1, color="gray", label="0.5 reference")
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.pdf", dpi=300)
    plt.close()

    if all_csv:
        csv_df = pd.concat(all_csv, ignore_index=True)
        csv_df.to_csv(f"{out_prefix}.csv", index=False)

    print(f"Saved {out_prefix}.pdf and {out_prefix}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiments", nargs="+",
        help="Alternating: tables_dir label tables_dir label ...",
    )
    parser.add_argument("-o", "--output", type=str, default="paper_win_rate",
                        help="Output file prefix (default: paper_win_rate)")
    parser.add_argument("--min-count", type=int, default=20,
                        help="Minimum observations per bin (default: 20)")
    parser.add_argument("--mass-bins", type=int, default=5,
                        help="Number of mass bins for random-null plots (default: 5)")
    args = parser.parse_args()

    if len(args.experiments) % 2 != 0:
        print("Error: experiments must be pairs of: tables_dir label", file=sys.stderr)
        sys.exit(1)

    experiments: List[Tuple[pathlib.Path, str]] = []
    for i in range(0, len(args.experiments), 2):
        experiments.append((pathlib.Path(args.experiments[i]), args.experiments[i + 1]))

    # --- Win rate by length (from null_queries.tsv) ---
    for qlab in ["target", "decoy"]:
        results = []
        for tables_dir, label in experiments:
            tsv = tables_dir / "null_queries.tsv"
            if not tsv.exists():
                print(f"Warning: {tsv} not found, skipping {label}", file=sys.stderr)
                continue
            df = pd.read_csv(tsv, sep="\t")
            sub = df[df["query_label"] == qlab].copy()
            rate_df = compute_win_rate_by_length(sub, "winner_cosine", args.min_count)
            results.append((label, rate_df))

        if results:
            plot_multi_experiment(
                results,
                x_col="query_length",
                xlabel="Query peptide length",
                ylabel="Target-win rate",
                title=f"Null cosine target-win rate by length ({qlab} queries)",
                out_prefix=f"{args.output}_by_length_{qlab}",
            )

    # --- Win rate by mass for random null queries (from random_null_queries.tsv) ---
    random_results = []
    for tables_dir, label in experiments:
        tsv = tables_dir / "random_null_queries.tsv"
        if not tsv.exists():
            print(f"Warning: {tsv} not found, skipping {label} for random null", file=sys.stderr)
            continue
        df = pd.read_csv(tsv, sep="\t")
        rate_df = compute_win_rate_by_mass(df, "winner_cosine", "query_mass", args.min_count, args.mass_bins)
        random_results.append((label, rate_df))

    if random_results:
        plot_multi_experiment(
            random_results,
            x_col="mass_center",
            xlabel="Query peptide mass (Da)",
            ylabel="Target-win rate",
            title="Random null cosine target-win rate by mass",
            out_prefix=f"{args.output}_random_by_mass",
        )


if __name__ == "__main__":
    main()
