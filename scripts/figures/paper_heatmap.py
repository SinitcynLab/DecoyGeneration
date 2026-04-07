#!/usr/bin/env python3
"""
Redraw nearest_decoy_cosine_distance_distribution_by_length_relative heatmap
from the diagnostics tables directory, with paper-quality formatting.

Usage:
    python paper_heatmap.py /path/to/tables_dir -o output_prefix

Outputs:
    {output_prefix}.pdf   – the heatmap figure
    {output_prefix}.csv   – raw data sufficient to reproduce the plot
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables_dir", type=pathlib.Path, help="Path to the diagnostics tables/ folder")
    parser.add_argument("-o", "--output", type=str, default="nearest_decoy_cosine_distance_heatmap",
                        help="Output file prefix (default: nearest_decoy_cosine_distance_heatmap)")
    parser.add_argument("--vmax", type=float, default=0.3, help="Max proportion for color scale (default: 0.3)")
    parser.add_argument("--nbins", type=int, default=40, help="Number of bins for cosine distance axis (default: 40)")
    args = parser.parse_args()

    tsv_path = args.tables_dir / "nonstealability.tsv"
    if not tsv_path.exists():
        print(f"Error: {tsv_path} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(tsv_path, sep="\t", usecols=["query_length", "best_decoy_cosine_distance"])
    df = df.dropna(subset=["query_length", "best_decoy_cosine_distance"])
    df = df[np.isfinite(df["best_decoy_cosine_distance"])]
    df["query_length"] = df["query_length"].astype(int)

    metric_edges = np.linspace(0.0, 1.0, args.nbins + 1)
    length_min = int(df["query_length"].min())
    length_max = int(df["query_length"].max())
    length_edges = np.arange(length_min, length_max + 2) - 0.5

    # histogram2d: x=query_length, y=cosine_distance
    hist, _, _ = np.histogram2d(
        df["query_length"].to_numpy(dtype=float),
        df["best_decoy_cosine_distance"].to_numpy(dtype=float),
        bins=[length_edges, metric_edges],
    )
    # Normalize each row (each length) to get proportions
    row_sums = hist.sum(axis=1, keepdims=True)
    rel = np.divide(hist, row_sums, out=np.zeros_like(hist), where=row_sums > 0)

    # --- Save CSV ---
    # Rows: one per (length, distance_bin) cell
    lengths = np.arange(length_min, length_max + 1)
    metric_centers = 0.5 * (metric_edges[:-1] + metric_edges[1:])
    csv_rows = []
    for i, length in enumerate(lengths):
        for j, center in enumerate(metric_centers):
            csv_rows.append({
                "query_length": int(length),
                "cosine_distance_bin_lo": float(metric_edges[j]),
                "cosine_distance_bin_hi": float(metric_edges[j + 1]),
                "cosine_distance_bin_center": float(center),
                "count": int(hist[i, j]),
                "proportion": float(rel[i, j]),
            })
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(f"{args.output}.csv", index=False)

    # --- Plot ---
    # Non-linear (power-law) norm to emphasize [0, 0.1] range
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=args.vmax)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Swapped axes: X = query_length, Y = cosine_distance
    mesh = ax.pcolormesh(
        length_edges, metric_edges, rel.T,
        shading="auto", norm=norm, cmap="viridis",
    )
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("Proportion within length")
    ax.set_xlabel("Query peptide length")
    ax.set_ylabel("Nearest-decoy cosine distance")
    ax.set_title("Nearest-decoy cosine distance by target length (relative)")

    plt.tight_layout()
    plt.savefig(f"{args.output}.pdf", dpi=300)
    plt.close()
    print(f"Saved {args.output}.pdf and {args.output}.csv")


if __name__ == "__main__":
    main()
