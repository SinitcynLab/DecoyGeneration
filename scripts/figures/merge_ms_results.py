#!/usr/bin/env python3
"""
Merge results from multiple ms_pipeline experiments into comparative plots and CSVs.

Usage:
    python merge_ms_results.py -o /path/to/output \\
        /path/to/exp1/output exp1 \\
        /path/to/exp2/output exp2

Generates both SVG and PDF for every plot.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLOT_EXTS = [".svg", ".pdf"]


def save_fig(fig, out_stem: Path, **kwargs) -> None:
    """Save figure as both SVG and PDF."""
    for ext in PLOT_EXTS:
        fig.savefig(out_stem.with_suffix(ext), **kwargs)


# Distinct colors for up to 10 experiments; cycles if more.
EXPERIMENT_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def get_color(idx: int) -> str:
    return EXPERIMENT_COLORS[idx % len(EXPERIMENT_COLORS)]


# ---------------------------------------------------------------------------
# CSV discovery & loading
# ---------------------------------------------------------------------------

TOOL_LEVEL_COMBOS = [
    ("sage", "psm"),
    ("sage", "peptide"),
    ("ms2rescore", "psm"),
    ("ms2rescore", "peptide"),
    ("oktoberfest", "psm"),
    ("oktoberfest", "peptide"),
]

CSV_KINDS = [
    "counts_vs_q",
    "entrapment_bounds_vs_q",
    "counts_vs_q_by_length",
]


def csv_name(tool: str, level: str, kind: str) -> str:
    return f"{tool}_{level}_{kind}.csv"


def load_csv(exp_dir: Path, tool: str, level: str, kind: str) -> Optional[pd.DataFrame]:
    path = exp_dir / "analysis" / csv_name(tool, level, kind)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def discover_available(experiments: Dict[str, Path]) -> List[Tuple[str, str, str]]:
    """Return (tool, level, kind) triples that exist in at least one experiment."""
    available = []
    for tool, level in TOOL_LEVEL_COMBOS:
        for kind in CSV_KINDS:
            for exp_dir in experiments.values():
                if (exp_dir / "analysis" / csv_name(tool, level, kind)).exists():
                    available.append((tool, level, kind))
                    break
    return available


def merge_csvs(
    experiments: Dict[str, Path], tool: str, level: str, kind: str,
) -> Optional[pd.DataFrame]:
    frames = []
    for name, exp_dir in experiments.items():
        df = load_csv(exp_dir, tool, level, kind)
        if df is not None:
            df.insert(0, "experiment", name)
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting: counts vs q
# ---------------------------------------------------------------------------

def plot_merged_counts_vs_q(
    merged: pd.DataFrame,
    out_path: Path,
    title: str,
    exp_names: List[str],
    ymax: float = 0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, name in enumerate(exp_names):
        df = merged[(merged["experiment"] == name) & (merged["q_threshold"] > 0)]
        if df.empty:
            continue
        ax.plot(
            df["q_threshold"], df["n_target_original"],
            label=name, color=get_color(idx), linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("q-value threshold")
    ax.set_ylabel("Count (original target)")
    ax.set_title(title)
    if ymax > 0:
        ax.set_ylim(0, ymax)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting: counts vs q by length
# ---------------------------------------------------------------------------

def plot_merged_counts_vs_q_by_length(
    merged: pd.DataFrame,
    out_path: Path,
    title: str,
    exp_names: List[str],
    ymax: float = 0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_lengths = sorted(merged["peptide_length"].dropna().unique().astype(int))
    if not all_lengths:
        return

    n = len(all_lengths)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for li, length in enumerate(all_lengths):
        ax = axes[li // ncols][li % ncols]
        for idx, name in enumerate(exp_names):
            df = merged[
                (merged["experiment"] == name)
                & (merged["peptide_length"] == length)
                & (merged["q_threshold"] > 0)
            ]
            if df.empty:
                continue
            ax.plot(
                df["q_threshold"], df["n_target_original"],
                label=name, color=get_color(idx), linewidth=1.2,
            )
        ax.set_xscale("log")
        ax.set_xlabel("q-value threshold")
        ax.set_ylabel("Count")
        ax.set_title(f"Length {length}")
        if ymax > 0:
            ax.set_ylim(0, ymax)
        ax.legend(fontsize=6)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting: entrapment bounds (shadow bands)
# ---------------------------------------------------------------------------

def plot_merged_entrapment_bounds(
    merged: pd.DataFrame,
    out_path: Path,
    title: str,
    exp_names: List[str],
) -> None:
    """
    Overlay entrapment FDP bounds as semi-transparent bands.

    Each experiment is drawn as:
      - a filled band between lower_bound_fdp and combined_upper_bound_fdp
      - a solid line for the upper bound (carries the legend label)
      - a dashed line for the lower bound

    Fill alpha scales down with the number of experiments so heavily
    overlapping bands remain readable.  The edge lines stay fully opaque
    so individual experiments are always traceable.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    n_exp = len(exp_names)
    fill_alpha = min(0.25, 0.6 / max(n_exp, 1))

    for idx, name in enumerate(exp_names):
        df = merged[(merged["experiment"] == name) & (merged["q_threshold"] > 0)].copy()
        if df.empty:
            continue
        df = df.sort_values("q_threshold")
        color = get_color(idx)

        q = df["q_threshold"].values
        lower = df["lower_bound_fdp"].values
        upper = df["combined_upper_bound_fdp"].values

        # fill_between: matplotlib silently clips non-positive values on log axes
        ax.fill_between(q, lower, upper, alpha=fill_alpha, color=color, linewidth=0)

        # Edge lines (mask zeros so they don't pull the log axis)
        upper_m = np.where(upper > 0, upper, np.nan)
        lower_m = np.where(lower > 0, lower, np.nan)
        ax.plot(q, upper_m, color=color, linewidth=1.5, linestyle="-", label=name)
        ax.plot(q, lower_m, color=color, linewidth=1.0, linestyle="--", alpha=0.7)

    # y = x reference
    q_ref = np.logspace(-3, -1, 100)
    ax.plot(q_ref, q_ref, "k--", linewidth=0.8, alpha=0.5, label="y = x")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e-1)
    ax.set_ylim(1e-3, 1e-1)
    ax.set_xlabel("FDR / q-value threshold")
    ax.set_ylabel("Estimated FDP")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------

TOOL_DISPLAY = {
    "sage": "Sage",
    "ms2rescore": "MS2Rescore",
    "oktoberfest": "Oktoberfest",
}

LEVEL_DISPLAY = {
    "psm": "PSM",
    "peptide": "Peptide",
}


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_experiments(args: List[str]) -> Dict[str, Path]:
    experiments: Dict[str, Path] = {}
    if len(args) % 2 != 0:
        print("Error: experiments must be given as pairs: /path/to/output name", file=sys.stderr)
        sys.exit(1)
    for i in range(0, len(args), 2):
        path_str, name = args[i], args[i + 1]
        path = Path(path_str)
        if not path.exists():
            print(f"Error: directory does not exist: {path}", file=sys.stderr)
            sys.exit(1)
        if not (path / "analysis").exists():
            print(f"Warning: no analysis/ in {path}, skipping", file=sys.stderr)
            continue
        experiments[name] = path
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge ms_pipeline results across experiments",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--ymax", type=float, default=0,
                        help="Max y-axis value for count vs q plots "
                             "(default: 0, auto)")
    parser.add_argument("experiments", nargs="+", help="/path/to/output name pairs")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = parse_experiments(args.experiments)
    if not experiments:
        print("Error: no valid experiments", file=sys.stderr)
        sys.exit(1)

    exp_names = list(experiments.keys())
    print(f"Merging {len(experiments)} experiments: {', '.join(exp_names)}")

    available = discover_available(experiments)
    if not available:
        print("No matching CSV files found", file=sys.stderr)
        sys.exit(1)

    for tool, level, kind in available:
        merged = merge_csvs(experiments, tool, level, kind)
        if merged is None or merged.empty:
            continue

        tool_d = TOOL_DISPLAY.get(tool, tool)
        level_d = LEVEL_DISPLAY.get(level, level)
        prefix = f"{tool}_{level}"

        # Save merged CSV
        csv_path = out_dir / f"{prefix}_{kind}.csv"
        merged.to_csv(csv_path, index=False)

        # Generate plots (both SVG and PDF via save_fig)
        # Pass a stem path; save_fig appends .svg and .pdf
        if kind == "counts_vs_q":
            plot_merged_counts_vs_q(
                merged,
                out_dir / f"{prefix}_counts_vs_q",
                title=f"{tool_d} {level_d} counts vs q",
                exp_names=exp_names,
                ymax=args.ymax,
            )
        elif kind == "counts_vs_q_by_length":
            plot_merged_counts_vs_q_by_length(
                merged,
                out_dir / f"{prefix}_counts_vs_q_by_length",
                title=f"{tool_d} {level_d} counts vs q by length",
                exp_names=exp_names,
                ymax=args.ymax,
            )
        elif kind == "entrapment_bounds_vs_q":
            plot_merged_entrapment_bounds(
                merged,
                out_dir / f"{prefix}_entrapment_bounds",
                title=f"{tool_d} {level_d} entrapment FDP bounds",
                exp_names=exp_names,
            )

        print(f"  {prefix}_{kind}")

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
