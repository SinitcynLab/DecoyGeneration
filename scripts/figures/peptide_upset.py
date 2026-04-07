#!/usr/bin/env python3
"""
Build an UpSet plot from peptide list files (one peptide per line),
optionally highlighting the top-k intersections and drawing sequence logos.

Usage:
    python peptide_upset.py -o upset.pdf -k 5 \
        peptides1.txt exp1 \
        peptides2.txt exp2 \
        peptides3.txt exp3

Outputs:
    <output>.pdf/.png  - the plot (format chosen by extension)
    <output_stem>.csv  - membership table (one row per peptide, boolean
                         columns per experiment) sufficient to recreate
                         the plot

Requires: upsetplot, logomaker, matplotlib, pandas, Pillow (for PNG combining)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from upsetplot import UpSet


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _build_pfm(peptides: list[str]) -> pd.DataFrame:
    """Build a position frequency matrix from variable-length peptides.

    Each position is normalised independently using only peptides that
    extend to that position.
    """
    if not peptides:
        return pd.DataFrame(columns=AMINO_ACIDS)
    max_len = max(len(p) for p in peptides)
    matrix = pd.DataFrame(0.0, index=range(max_len), columns=AMINO_ACIDS)
    for pos in range(max_len):
        counts: dict[str, int] = {}
        total = 0
        for p in peptides:
            if pos < len(p) and p[pos] in AMINO_ACIDS:
                counts[p[pos]] = counts.get(p[pos], 0) + 1
                total += 1
        if total > 0:
            for aa in AMINO_ACIDS:
                matrix.loc[pos, aa] = counts.get(aa, 0) / total
    return matrix


def _peptides_for_subset(all_peptides: list[str],
                         membership: pd.DataFrame,
                         subset_key: tuple[bool, ...]) -> list[str]:
    """Return peptides belonging to exactly the given intersection."""
    mask = pd.Series(True, index=membership.index)
    for col, val in zip(membership.columns, subset_key):
        mask &= membership[col] == val
    return [all_peptides[i] for i in range(len(all_peptides)) if mask.iloc[i]]


def _save_csv(out_path: Path, all_peptides: list[str],
              membership: pd.DataFrame) -> None:
    """Write a CSV with one row per peptide and boolean membership columns."""
    df = membership.copy()
    df.insert(0, "peptide", all_peptides)
    df.to_csv(out_path, index=False)
    print(f"  CSV → {out_path}")


def _build_upset_fig(series, names, top_k_keys, colors, caption=""):
    """Build and return the UpSet matplotlib figure."""
    upset = UpSet(series, show_counts=True, sort_by="cardinality")
    if top_k_keys is not None:
        for idx, color in zip(top_k_keys, colors):
            present = [n for n, v in zip(names, idx) if v]
            absent = [n for n, v in zip(names, idx) if not v]
            upset.style_subsets(present=present, absent=absent, facecolor=color)

    fig = plt.figure(figsize=(3 + 2 * len(names), 8))
    upset.plot(fig=fig)
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic")
    return fig


def _build_logo_fig(top_k_keys, colors, names,
                    all_peptides, membership,
                    all_peptides_full, membership_full):
    """Build and return the logo-grid matplotlib figure."""
    k = len(top_k_keys)
    ncols = min(k, 4)
    nrows = (k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(5 * ncols, 3 * nrows))

    for idx_i, (subset_key, color) in enumerate(zip(top_k_keys, colors)):
        row, col = divmod(idx_i, ncols)
        ax = axes[row][col]

        peps = _peptides_for_subset(all_peptides, membership, subset_key)
        peps_full = _peptides_for_subset(
            all_peptides_full, membership_full, subset_key)
        label_parts = [n for n, v in zip(names, subset_key) if v]
        label = " \u2229 ".join(label_parts) if label_parts else "none"
        counts = (f"{len(peps_full)} total | {len(peps)} filtered"
                  if len(peps_full) != len(peps)
                  else f"{len(peps)} total")

        if len(peps) < 2:
            ax.text(0.5, 0.5,
                    f"{label}\n({counts})\ntoo few for logo",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            pfm = _build_pfm(peps)
            info = logomaker.transform_matrix(pfm, from_type="probability",
                                              to_type="information")
            logomaker.Logo(info, ax=ax)
            ax.set_ylim(0, 4.5)
            ax.set_ylabel("bits", fontsize=8)
            ax.set_xlabel("position", fontsize=8)

        ax.set_title(f"{label}\n{counts}", fontsize=9, fontweight="bold",
                     color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    for idx_i in range(k, nrows * ncols):
        row, col = divmod(idx_i, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig


def _save_pdf(out_path: str, figures: list):
    """Write all figures as pages in a single PDF."""
    with PdfPages(out_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    plt.close("all")


def _save_image(out_path: str, figures: list):
    """Render figures to temp PNGs, stack vertically, save as image."""
    from PIL import Image

    tmp_paths = []
    images = []
    for i, fig in enumerate(figures):
        tmp = out_path + f".tmp_{i}.png"
        fig.savefig(tmp, dpi=150, bbox_inches="tight")
        tmp_paths.append(tmp)
        images.append(Image.open(tmp))
    plt.close("all")

    total_w = max(img.width for img in images)
    total_h = sum(img.height for img in images)
    combined = Image.new("RGB", (total_w, total_h), "white")
    y = 0
    for img in images:
        combined.paste(img, (0, y))
        y += img.height
    combined.save(out_path, dpi=(150, 150))

    for tmp in tmp_paths:
        Path(tmp).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an UpSet plot from peptide list files",
    )
    parser.add_argument("-o", "--output", required=True,
                        help="Output plot file (.pdf or .png)")
    parser.add_argument("-k", "--top-k", type=int, default=0,
                        help="Highlight top-k intersections and draw logos "
                             "(default: 0, no logos)")
    parser.add_argument("-l", "--length", type=int, default=0,
                        help="Keep only peptides of this length "
                             "(default: 0, keep all)")
    parser.add_argument("inputs", nargs="+",
                        help="Pairs of: peptides.txt name")
    args = parser.parse_args()

    if len(args.inputs) % 2 != 0:
        print("Error: arguments must be pairs: file1 name1 file2 name2 ...",
              file=sys.stderr)
        sys.exit(1)

    # Load experiments (all peptides for UpSet, filtered for logos)
    experiments: list[tuple[str, set[str]]] = []
    experiments_filtered: list[tuple[str, set[str]]] = []
    for i in range(0, len(args.inputs), 2):
        path = Path(args.inputs[i])
        name = args.inputs[i + 1]
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        peptides = set(path.read_text().strip().splitlines())
        experiments.append((name, peptides))
        if args.length > 0:
            filtered = {p for p in peptides if len(p) == args.length}
            experiments_filtered.append((name, filtered))
            print(f"  {name}: {len(peptides)} peptides ({len(filtered)} length={args.length})")
        else:
            experiments_filtered.append((name, peptides))
            print(f"  {name}: {len(peptides)} peptides")

    names = [name for name, _ in experiments]
    all_peptides = sorted(set().union(*(peps for _, peps in experiments)))
    all_peptides_filtered = sorted(
        set().union(*(peps for _, peps in experiments_filtered)))
    print(f"  Total unique: {len(all_peptides)}")

    if args.length > 0:
        caption = f"{len(all_peptides)} total | {len(all_peptides_filtered)} length={args.length}"
    else:
        caption = f"{len(all_peptides)} total"

    # Boolean membership DataFrame (rows = peptides, columns = experiments)
    # Full set for UpSet plot
    membership = pd.DataFrame(
        {name: [p in peps for p in all_peptides] for name, peps in experiments},
    )
    # Filtered set for logos
    membership_filtered = pd.DataFrame(
        {name: [p in peps for p in all_peptides_filtered]
         for name, peps in experiments_filtered},
    )

    # Write CSV alongside the plot
    out_path = Path(args.output)
    csv_path = out_path.with_suffix(".csv")
    _save_csv(csv_path, all_peptides, membership)

    # MultiIndex series for upsetplot (all peptides)
    mi_df = membership.copy()
    mi_df.index = all_peptides
    mi_df = mi_df.set_index(names)
    series = mi_df.groupby(level=names).size()

    k = args.top_k

    # Determine top-k keys and colours (None when k <= 0)
    top_k_keys = None
    colors = []
    if k > 0:
        top_k_keys = series.sort_values(ascending=False).head(k).index.tolist()
        if len(names) == 1:
            top_k_keys = [(v,) for v in top_k_keys]
        colors = plt.cm.tab10(np.linspace(0, 1, max(k, 10)))[:k]

    # Build figures
    figures = [_build_upset_fig(series, names, top_k_keys, colors, caption)]
    if k > 0:
        figures.append(
            _build_logo_fig(top_k_keys, colors, names,
                            all_peptides_filtered, membership_filtered,
                            all_peptides, membership))

    # Save
    if out_path.suffix.lower() == ".pdf":
        _save_pdf(args.output, figures)
    else:
        if len(figures) == 1:
            figures[0].savefig(args.output, dpi=150, bbox_inches="tight")
            plt.close("all")
        else:
            _save_image(args.output, figures)

    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
