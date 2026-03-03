import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Optional

import re
import math

from .util import split_proteins_field


def plot_counts_vs_q(counts_df: pd.DataFrame, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Avoid log(0) issues
    df = counts_df[counts_df["q_threshold"] > 0].copy()
    plt.figure()
    plt.plot(df["q_threshold"], df["n_rows"], label="All accepted")
    plt.plot(df["q_threshold"], df["n_target_original"], label="Original target")
    plt.plot(df["q_threshold"], df["n_entrapment"], label="Entrapment")
    plt.xscale("log")
    plt.xlabel("q-value threshold")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_entrapment_bounds(bounds_df: pd.DataFrame, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Avoid log(0) issues
    df = bounds_df[bounds_df["q_threshold"] > 0].copy()
    plt.figure()
    plt.plot(df["q_threshold"], df["lower_bound_fdp"], label="Lower bound (entrapment)")
    plt.plot(df["q_threshold"], df["combined_upper_bound_fdp"], label="Combined upper bound (entrapment)")
    plt.plot(df["q_threshold"], df["q_threshold"], label="y = x (reference)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FDR/q-value threshold used by tool")
    plt.ylabel("Estimated FDP")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


_CATEGORY_COLORS = {
    "normal_target": "#1f77b4",       # blue
    "normal_decoy": "#aec7e8",        # light blue
    "entrapment_target": "#d62728",   # red
    "entrapment_decoy": "#ff9896",    # light red / pink
}

_CATEGORY_ORDER = ["normal_target", "normal_decoy", "entrapment_target", "entrapment_decoy"]


def _add_4way_category(
    df: pd.DataFrame,
    proteins_col: str,
    entrapment_prefix: str,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Add '_category' column with values:
      normal_target / normal_decoy / entrapment_target / entrapment_decoy.
    """
    out = df.copy()

    def _is_ent(prot_str):
        prots = split_proteins_field(str(prot_str) if pd.notna(prot_str) else "")
        prots_clean = [re.sub(r"^(rev_|decoy_|DECOY_)", "", p) for p in prots]
        # Use "unambiguous" strategy (same as label_entrapment_rows): ALL proteins must be
        # entrapment. PSMs shared between entrapment and target proteins (e.g. conserved
        # proteins like EF1A) stay in normal_target / normal_decoy.
        return bool(prots_clean) and all(p.startswith(entrapment_prefix) for p in prots_clean)

    is_ent = out[proteins_col].apply(_is_ent)
    is_decoy = pd.to_numeric(out[label_col], errors="coerce") == -1

    out["_category"] = "normal_target"
    out.loc[is_decoy & ~is_ent, "_category"] = "normal_decoy"
    out.loc[~is_decoy & is_ent, "_category"] = "entrapment_target"
    out.loc[is_decoy & is_ent, "_category"] = "entrapment_decoy"
    return out


def plot_score_distributions(
    df: pd.DataFrame,
    score_col: str,
    category_col: str,
    out_png: Path,
    title: str,
    discrete: bool,
    x_name: str,
    absolute: bool,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for cat in _CATEGORY_ORDER:
        subset = df[df[category_col] == cat][score_col].dropna()
        n = len(subset)
        if n > 1:
            if discrete:
                value_counts = subset.value_counts().sort_index()
                ax.plot(
                    value_counts.index,
                    value_counts.values,
                    marker="o",
                    linestyle="-",
                    label=f"{cat} (N={n:,})",
                    color=_CATEGORY_COLORS[cat],
                )
            else:
                counts, bins = np.histogram(subset, bins=40, density=not absolute)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                ax.plot(bin_centers, counts, label=f"{cat} (N={n:,})", color=_CATEGORY_COLORS[cat])
        else:
            ax.plot([], [], color=_CATEGORY_COLORS[cat], label=f"{cat} (N=0)")

    ax.set_xlabel(x_name)
    if discrete:
        ax.set_ylabel("Count")
    else:
        ax.set_ylabel("Count" if absolute else "Proportion")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_score_distributions_by_length(
    df: pd.DataFrame,
    score_col: str,
    category_col: str,
    length_col: str,
    out_png: Path,
    title: str,
    discrete: bool,
    x_name: str,
    absolute: bool,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    lengths = sorted(df[length_col].dropna().unique())
    if not lengths:
        return

    n_lengths = len(lengths)
    ncols = min(4, n_lengths)
    nrows = math.ceil(n_lengths / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, length in enumerate(lengths):
        ax = axes[idx // ncols][idx % ncols]
        subset = df[df[length_col] == length]

        for cat in _CATEGORY_ORDER:
            cat_data = subset[subset[category_col] == cat][score_col].dropna()
            n = len(cat_data)
            if n > 1:
                if discrete:
                    value_counts = cat_data.value_counts().sort_index()
                    ax.plot(
                        value_counts.index,
                        value_counts.values,
                        marker="o",
                        linestyle="-",
                        label=f"{cat} (N={n:,})",
                        color=_CATEGORY_COLORS[cat],
                    )
                else:
                    counts, bins = np.histogram(cat_data, bins=40, density=not absolute)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    ax.plot(bin_centers, counts, label=f"{cat} (N={n:,})", color=_CATEGORY_COLORS[cat])
            else:
                ax.plot([], [], color=_CATEGORY_COLORS[cat], label=f"{cat} (N=0)")

        ax.set_title(f"Length {int(length)}")
        ax.set_xlabel(x_name)
        if discrete:
            ax.set_ylabel("Count")
        else:
            ax.set_ylabel("Density" if absolute else "Proportion")
        ax.set_yscale("log")
        ax.legend(fontsize=6)

    for idx in range(n_lengths, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def run_length_distribution_plots(
    df: pd.DataFrame,
    *,
    score_col: str,
    plot_name: str,
    short_name: str,
    proteins_col: str,
    label_col: str,
    entrapment_prefix: str,
    peptide_col: str,
    out_dir: Path,
    file_prefix: str,
    title_prefix: str,
    experiment_name: str,
    discrete: bool,
) -> None:
    if score_col not in df.columns:
        return
    
    work = df.copy()
    work = _add_4way_category(work, proteins_col, entrapment_prefix, label_col)

    def _peptide_length(seq) -> Optional[int]:
        if pd.isna(seq) or str(seq).strip() == "":
            return None
        # Remove modifications
        clean = re.sub(r"[^A-Za-z]", "", str(seq))
        return len(clean) if clean else None
    work["_peptide_length"] = work[peptide_col].apply(_peptide_length)

    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[score_col])
    if work.empty:
        return

    if not discrete:
        plot_score_distributions(
            work, score_col, "_category",
            out_dir / f"{file_prefix}_{short_name}_dist.png",
            f"{title_prefix} {plot_name} Distribution ({experiment_name})",
            discrete=discrete, x_name=plot_name, absolute=False,
        )
    plot_score_distributions(
        work, score_col, "_category",
        out_dir / f"{file_prefix}_{short_name}_abs_dist.png",
        f"{title_prefix} {plot_name} Distribution ({experiment_name})",
        discrete=discrete, x_name=plot_name, absolute=True,
    )

    if work["_peptide_length"].notna().any():
        if not discrete:
            plot_score_distributions_by_length(
                work, score_col, "_category", "_peptide_length",
                out_dir / f"{file_prefix}_{short_name}_by_length.png",
                f"{title_prefix} {plot_name} Distribution by Peptide Length ({experiment_name})",
                discrete=discrete, x_name=plot_name, absolute=False,
            ) 
        plot_score_distributions_by_length(
            work, score_col, "_category", "_peptide_length",
            out_dir / f"{file_prefix}_{short_name}_by_length_abs.png",
            f"{title_prefix} {plot_name} Distribution by Peptide Length ({experiment_name})",
            discrete=discrete, x_name=plot_name, absolute=True,
        )

def run_all_length_distribution_plots(
    df: pd.DataFrame,
    *,
    proteins_col: str,
    label_col: str,
    entrapment_prefix: str,
    peptide_col: str,
    out_dir: Path,
    file_prefix: str,
    title_prefix: str,
    experiment_name: str,
) -> None:
    run_length_distribution_plots(
        df, score_col="_score", plot_name="Score", short_name="score",
        proteins_col=proteins_col, label_col=label_col, entrapment_prefix=entrapment_prefix, peptide_col=peptide_col,
        out_dir=out_dir, file_prefix=file_prefix, title_prefix=title_prefix, experiment_name=experiment_name, discrete=False,
    )
    run_length_distribution_plots(
        df, score_col="matched_peaks", plot_name="Matched Peaks", short_name="matched",
        proteins_col=proteins_col, label_col=label_col, entrapment_prefix=entrapment_prefix, peptide_col=peptide_col,
        out_dir=out_dir, file_prefix=file_prefix, title_prefix=title_prefix, experiment_name=experiment_name, discrete=True,
    )
    run_length_distribution_plots(
        df, score_col="longest_b", plot_name="Longest b-ion series", short_name="longest_b",
        proteins_col=proteins_col, label_col=label_col, entrapment_prefix=entrapment_prefix, peptide_col=peptide_col,
        out_dir=out_dir, file_prefix=file_prefix, title_prefix=title_prefix, experiment_name=experiment_name, discrete=True,
    )
    run_length_distribution_plots(
        df, score_col="longest_y", plot_name="Longest y-ion series", short_name="longest_y",
        proteins_col=proteins_col, label_col=label_col, entrapment_prefix=entrapment_prefix, peptide_col=peptide_col,
        out_dir=out_dir, file_prefix=file_prefix, title_prefix=title_prefix, experiment_name=experiment_name, discrete=True,
    )
