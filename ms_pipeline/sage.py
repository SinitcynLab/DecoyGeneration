from typing import Optional
from pathlib import Path

import json

import pandas as pd

from .entrapment import EntrapmentConfig


DEFAULT_SAGE_CONFIG = {
    "database": {
        "bucket_size": 8192,
        "enzyme": {
            "missed_cleavages": 2,
            "min_len": 7,
            "max_len": 50,
            "cleave_at": "KR",
            "restrict": "P",
            "c_terminal": True,
            "semi_enzymatic": False,
        },
        "peptide_min_mass": 500.0,
        "peptide_max_mass": 5000.0,
        "ion_kinds": ["b", "y"],
        "min_ion_index": 2,
        "max_variable_mods": 0,
        "static_mods": {},
        "variable_mods": {},
        "decoy_tag": "rev_",
        "generate_decoys": False,
        "fasta": "REPLACED_BY_CLI",
    },
    "deisotope": True,
    "chimera": False,
    "max_fragment_charge": 2,
    "report_psms": 200,
    "precursor_tol": {"ppm": [-10, 10]},
    "fragment_tol": {"ppm": [-10, 10]},
    "isotope_errors": [-1, 3],
    "score_type": "SageHyperScore",
}


def load_or_create_sage_config(
    *,
    sage_config_in: Optional[Path],
    out_path: Path,
    entrap_cfg: EntrapmentConfig,
    fragmentation: str = "hcd",
    protease_cfg=None,
) -> Path:
    if sage_config_in is None:
        cfg = json.loads(json.dumps(DEFAULT_SAGE_CONFIG))
    else:
        cfg = json.loads(sage_config_in.read_text(encoding="utf-8"))

    # Set ion_kinds based on fragmentation type
    if fragmentation == "etd":
        cfg.setdefault("database", {})["ion_kinds"] = ["c", "z"]

    # Override enzyme based on protease
    if protease_cfg is not None:
        db = cfg.setdefault("database", {})
        enz = db.setdefault("enzyme", {})
        enz["cleave_at"] = protease_cfg.cleave_at
        enz["restrict"] = protease_cfg.restrict
        enz["c_terminal"] = protease_cfg.c_terminal
        if protease_cfg.min_len is not None:
            enz["min_len"] = protease_cfg.min_len
        if protease_cfg.max_len is not None:
            enz["max_len"] = protease_cfg.max_len

    if entrap_cfg.mode == "paired_shuffled_peptides":
        # ensure enzyme exists
        db = cfg.setdefault("database", {})
        enz = db.setdefault("enzyme", {})
        enz["cleave_at"] = entrap_cfg.paired_cleave_at_special_for_search  # "$" by default
        # still apply min/max len (those are checked even with "$")
        enz.setdefault("min_len", entrap_cfg.paired_min_len)
        enz.setdefault("max_len", entrap_cfg.paired_max_len)
        enz.setdefault("missed_cleavages", 0)

    # TODO: why does not work?
    if entrap_cfg.decoy_prefix:
        db = cfg.setdefault("database", {})
        db["decoy_tag"] = entrap_cfg.decoy_prefix
        db["generate_decoys"] = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out_path

def load_sage_results(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path, sep="\t", low_memory=False)
    # Normalize columns for downstream logic
    # q columns in Sage output are spectrum_q, peptide_q, protein_q
    # target/decoy label: either 'label' (1 target, -1 decoy) or 'is_decoy' bool
    if "label" not in df.columns and "is_decoy" in df.columns:
        df["label"] = df["is_decoy"].apply(lambda x: -1 if bool(x) else 1)
    # ensure numeric q cols
    for qc in ["spectrum_q", "peptide_q", "protein_q"]:
        if qc in df.columns:
            df[qc] = pd.to_numeric(df[qc], errors="coerce")
    # best score column for paired estimator comparisons
    if "sage_discriminant_score" in df.columns:
        df["_score"] = pd.to_numeric(df["sage_discriminant_score"], errors="coerce")
    elif "hyperscore" in df.columns:
        df["_score"] = pd.to_numeric(df["hyperscore"], errors="coerce")
    else:
        df["_score"] = pd.NA

    df["matched_peaks"] = pd.to_numeric(df["matched_peaks"], errors="coerce")
    df["longest_b"] = pd.to_numeric(df["longest_b"], errors="coerce")
    df["longest_y"] = pd.to_numeric(df["longest_y"], errors="coerce")

    return df

def aggregate_to_peptide_level_from_sage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a peptide-level table from Sage PSMs.

    We:
      - keep label==1 and label==-1 rows (so q values remain meaningful if you re-run a peptide-level procedure later)
      - aggregate by stripped_peptide if available, else peptide
      - take min peptide_q per peptide (or min spectrum_q as fallback)
      - keep a representative proteins string from the best q row
    """
    work = df.copy()
    pep_col = "stripped_peptide" if "stripped_peptide" in work.columns else "peptide"
    if pep_col not in work.columns:
        raise KeyError("Cannot aggregate: missing peptide/stripped_peptide column.")

    # Choose q column: prefer peptide_q; fallback spectrum_q
    if "peptide_q" in work.columns:
        qcol = "peptide_q"
    elif "spectrum_q" in work.columns:
        qcol = "spectrum_q"
    else:
        raise KeyError("Cannot aggregate: missing peptide_q and spectrum_q.")

    # Keep best row per peptide for proteins field and score
    work[qcol] = pd.to_numeric(work[qcol], errors="coerce")
    work["_score"] = pd.to_numeric(work.get("_score", pd.Series([pd.NA] * len(work))), errors="coerce")
    work["matched_peaks"] = pd.to_numeric(work.get("matched_peaks", pd.Series([pd.NA] * len(work))), errors="coerce")
    work["longest_b"] = pd.to_numeric(work.get("longest_b", pd.Series([pd.NA] * len(work))), errors="coerce")
    work["longest_y"] = pd.to_numeric(work.get("longest_y", pd.Series([pd.NA] * len(work))), errors="coerce")
    work = work.dropna(subset=[qcol])

    # idx of min q per peptide
    idx = work.groupby(pep_col)[qcol].idxmin()
    rep = work.loc[idx, [pep_col, qcol, "label", "proteins", "_score", "matched_peaks", "longest_b", "longest_y"]].copy()
    rep = rep.rename(columns={pep_col: "peptide", qcol: "q"})
    return rep
