#!/usr/bin/env python3
"""
Compare nearest-decoy cosine distance distributions at a fixed peptide length
across multiple experiments.

Usage:
    python paper_distance_by_length.py \\
        --length 10 \\
        /path/to/tables1 "Reverse" \\
        /path/to/tables2 "Shuffle" \\
        -o output_prefix

Each positional argument pair is: tables_dir label.

Outputs:
    {output_prefix}.pdf   – the figure
    {output_prefix}.csv   – raw data sufficient to reproduce the plot
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiments", nargs="+",
                        help="Alternating: tables_dir label tables_dir label ...")
    parser.add_argument("--length", type=int, required=True,
                        help="Peptide length to compare")
    parser.add_argument("-o", "--output", type=str,
                        default="paper_distance_by_length",
                        help="Output file prefix (default: paper_distance_by_length)")
    parser.add_argument("--nbins", type=int, default=40,
                        help="Number of bins for CSV output (default: 40)")
    parser.add_argument("--kde-points", type=int, default=200,
                        help="Number of evaluation points for KDE curve (default: 200)")
    parser.add_argument("--bootstrap", type=int, default=200,
                        help="Number of bootstrap resamples for confidence band (default: 200)")
    args = parser.parse_args()

    if len(args.experiments) % 2 != 0:
        print("Error: experiments must be pairs of: tables_dir label", file=sys.stderr)
        sys.exit(1)

    experiments: List[Tuple[pathlib.Path, str]] = []
    for i in range(0, len(args.experiments), 2):
        experiments.append((pathlib.Path(args.experiments[i]), args.experiments[i + 1]))

    x_plot = np.linspace(0.0, 1.0, args.kde_points)
    edges = np.linspace(0.0, 1.0, args.nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(9, 5))
    all_csv = []

    for tables_dir, label in experiments:
        tsv = tables_dir / "nonstealability.tsv"
        if not tsv.exists():
            print(f"Warning: {tsv} not found, skipping {label}", file=sys.stderr)
            continue
        df = pd.read_csv(tsv, sep="\t", usecols=["query_length", "best_decoy_cosine_distance"])
        df = df.dropna(subset=["query_length", "best_decoy_cosine_distance"])
        df = df[np.isfinite(df["best_decoy_cosine_distance"])]
        sub = df[df["query_length"] == args.length]
        if sub.empty:
            print(f"Warning: no data for length {args.length} in {label}", file=sys.stderr)
            continue

        vals = sub["best_decoy_cosine_distance"].to_numpy(dtype=float)
        kde = gaussian_kde(vals)
        density = kde(x_plot)
        line, = ax.plot(x_plot, density, label=label)
        color = line.get_color()

        # Bootstrap confidence band on the KDE
        rng = np.random.default_rng(42)
        boot_curves = np.empty((args.bootstrap, len(x_plot)))
        for b in range(args.bootstrap):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boot_curves[b] = gaussian_kde(sample)(x_plot)
        lo_density = np.percentile(boot_curves, 2.5, axis=0)
        hi_density = np.percentile(boot_curves, 97.5, axis=0)
        ax.fill_between(x_plot, lo_density, hi_density, alpha=0.15, color=color)

        # CSV: binned histogram (sufficient to reproduce)
        counts, _ = np.histogram(vals, bins=edges)
        total = counts.sum()
        proportion = counts / total if total > 0 else counts.astype(float)
        csv_part = pd.DataFrame({
            "bin_lo": edges[:-1],
            "bin_hi": edges[1:],
            "bin_center": centers,
            "count": counts,
            "proportion": proportion,
            "experiment": label,
        })
        all_csv.append(csv_part)

    ax.set_xlabel("Nearest-decoy cosine distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Nearest-decoy cosine distance distribution (length = {args.length})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{args.output}.pdf", dpi=300)
    plt.close()

    if all_csv:
        pd.concat(all_csv, ignore_index=True).to_csv(f"{args.output}.csv", index=False)

    print(f"Saved {args.output}.pdf and {args.output}.csv")


if __name__ == "__main__":
    main()
