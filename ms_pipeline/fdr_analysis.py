import numpy as np
import math
import pandas as pd
import re

from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


from .protease import digest_fasta_unique_peptides
from .entrapment import EntrapmentLabeling, label_entrapment_rows
from .util import split_proteins_field


def compute_r_effective(
    *,
    target_fasta: Path,
    entrapment_fasta: Path,
    enzyme_cleave_at: str,
    restrict: str,
    missed_cleavages: int,
    min_len: int,
    max_len: int,
    c_terminal: bool,
    collapse_il: bool,
) -> float:
    t_uniq, _ = digest_fasta_unique_peptides(
        target_fasta,
        enzyme_cleave_at=enzyme_cleave_at,
        restrict=restrict,
        missed_cleavages=missed_cleavages,
        min_len=min_len,
        max_len=max_len,
        c_terminal=c_terminal,
        collapse_il=collapse_il,
    )
    e_uniq, _ = digest_fasta_unique_peptides(
        entrapment_fasta,
        enzyme_cleave_at=enzyme_cleave_at,
        restrict=restrict,
        missed_cleavages=missed_cleavages,
        min_len=min_len,
        max_len=max_len,
        c_terminal=c_terminal,
        collapse_il=collapse_il,
    )
    return e_uniq / t_uniq

def analyze_counts_and_entrapment(
    df: pd.DataFrame,
    *,
    q_col: str,
    proteins_col: str,
    label_col: str = "label",
    labeling: Optional[EntrapmentLabeling] = None,
    r_effective: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Core analysis for one PSM table.

    Returns a dict of named DataFrames:
      - counts_vs_q
      - entrapment_bounds_vs_q
      - paired_upper_vs_q (optional)
    """
    if labeling is None:
        labeling = EntrapmentLabeling()

    q_min = 1e-3
    q_max = 0.1
    q_grid = list(np.unique(np.concatenate(([0.0], np.logspace(math.log10(q_min), math.log10(q_max), num=100), [q_max]))))

    work = df.copy()
    work[q_col] = pd.to_numeric(work[q_col], errors="coerce")
    work = work.dropna(subset=[q_col])
    if label_col not in work.columns:
        raise KeyError(f"Missing label column '{label_col}'")
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")

    work_targets = work[work[label_col] == 1].copy()

    # Label entrapment
    labeled = label_entrapment_rows(work_targets, proteins_col=proteins_col, labeling=labeling)

    # For peptide-level counts, users may pass pre-aggregated tables. Here we treat rows as units.
    # If you want peptide-level: pass a df already grouped to unique peptides (see helper below).
    rows: List[Dict[str, float]] = []
    bounds_rows: List[Dict[str, float]] = []

    # If r not provided, default to 1 so combined and lower differ by factor 2 at r=1.
    r = float(r_effective) if (r_effective is not None and r_effective > 0) else 1.0

    for t in q_grid:
        sel = labeled[labeled[q_col] <= t]
        n_all = int(sel.shape[0])
        n_e = int(sel["is_entrapment"].sum())
        n_t = int(sel["is_target_original"].sum())
        # Note: ambiguous entrapment rows are excluded from both n_e and n_t by construction above:
        # - is_entrapment requires all entrapment (unambiguous default)
        # - is_target_original requires none entrapment
        # If a row is ambiguous, it contributes neither. This is intentional but configurable.
        n_amb = int(sel["is_ambiguous_entrapment"].sum())

        rows.append({"q_threshold": float(t), "n_rows": n_all, "n_target_original": n_t, "n_entrapment": n_e, "n_ambiguous": n_amb})

        lower = n_e / (n_t + n_e) if (n_t + n_e) > 0 else 0.0
        combined = (n_e * (1.0 + (1.0 / r))) / (n_t + n_e) if (n_t + n_e) > 0 else 0.0
        bounds_rows.append({"q_threshold": float(t), "lower_bound_fdp": lower, "combined_upper_bound_fdp": combined, "r_effective": r})

    out: Dict[str, pd.DataFrame] = {
        "counts_vs_q": pd.DataFrame(rows),
        "entrapment_bounds_vs_q": pd.DataFrame(bounds_rows),
    }
    return out
