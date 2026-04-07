#!/usr/bin/env python3
"""
Oktoberfest rescoring integration.

This module provides:
  - a small runner helper that writes an Oktoberfest config JSON and executes:
        python -m oktoberfest --config_path <config.json>
  - robust, search-engine-agnostic parsing helpers for the produced
    mokapot/percolator PSM output files:
        results/<method>/rescore.<method>.psms.txt
        results/<method>/rescore.<method>.decoy.psms.txt

Notes:
  - Oktoberfest expects *unfiltered* search results (100% FDR) for rescoring.
  - When integrating into your pipeline, keep in mind that Oktoberfest's "output"
    path in config is interpreted relative to the config file location.

This is intentionally a "stub" / integration shim:
  - you can pass a fully custom config path and skip config generation
  - you can customize extra config keys via `extra_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import json

import pandas as pd

from .cmd import run_cmd


def _read_table_guess(path: Path) -> pd.DataFrame:
    """
    Read a percolator/mokapot style table.

    Oktoberfest outputs are typically tab-delimited .txt files; we try tab first,
    then fall back to pandas' separator inference.
    """
    try:
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        # If it parsed into a single wide column, fallback to sep inference
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=None, engine="python", comment="#", low_memory=False)
        return df
    except Exception:
        # last-resort: separator inference
        return pd.read_csv(path, sep=None, engine="python", comment="#", low_memory=False)


def _find_col(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c2 = lower_map.get(cand.lower())
        if c2 is not None:
            return c2
    if required:
        raise KeyError(f"None of candidate columns found: {candidates}. Columns: {list(df.columns)[:80]} ...")
    return None


@dataclass
class OktoberfestRescoreConfig:
    """
    Minimal config needed for an Oktoberfest rescoring job.

    You can set `config_path` to an existing config JSON and skip generation.
    Otherwise we write the config to: <work_dir>/oktoberfest_config.json
    """
    work_dir: Path
    search_results: Path
    spectra: Path

    # These map to Oktoberfest config keys (see docs).
    search_results_type: str = "Sage"        # "Maxquant" | "Msfragger" | "Sage" | ... | "Internal"
    spectra_type: str = "mzml"              # "raw" | "mzml" | "d" | "hdf"
    instrument_type: str = "QE"             # "QE" | "LUMOS" | "TIMSTOF" | "SCIEXTOF"
    tag: str = ""                           # e.g. "tmt", "tmtpro", "itraq4", "itraq8"

    # Models / prediction server
    intensity_model: str = "Prosit_2020_intensity_HCD"
    irt_model: str = "Prosit_2019_irt"
    prediction_server: str = "koina.wilhelmlab.org:443"
    ssl: bool = True

    # Rescoring / FDR estimation
    fdr_estimation_method: str = "percolator"   # "mokapot" | "percolator"
    regressionMethod: str = "spline"         # "spline" | "lowess" | "logistic"
    add_feature_cols: Union[str, List[str]] = "none"  # "none" | "all" | [col1, col2, ...]

    # IO / resources
    numThreads: int = 16                      # parallelisation on file level
    thermoExe: Optional[str] = None          # needed if spectra_type == "raw"

    # Peak annotation tolerance
    massTolerance: float = 20.0
    unitMassTolerance: str = "ppm"           # "da" | "ppm"

    # CE calibration options
    ce_range: Tuple[int, int] = (19, 50)
    use_ransac_model: bool = False

    # Optional custom mod handling
    static_mods: Optional[Dict[str, List[float]]] = None
    var_mods: Optional[Dict[str, List[float]]] = None

    # Advanced: arbitrary additional keys merged into root config dict
    extra_config: Dict[str, object] = field(default_factory=dict)

    # Advanced: if set, use this config file and don't auto-generate
    config_path: Optional[Path] = None

    # Which python executable to run with
    python_bin: str = "python"

    # Where Oktoberfest should write outputs; relative to config location.
    # We default to "." so results land under work_dir directly.
    output_rel: str = "."

    def to_dict(self) -> Dict[str, object]:
        cfg: Dict[str, object] = {
            "type": "Rescoring",
            "tag": self.tag,
            "output": self.output_rel,
            "inputs": {
                "search_results": str(self.search_results),
                "search_results_type": self.search_results_type,
                "spectra": str(self.spectra),
                "spectra_type": self.spectra_type,
                "instrument_type": self.instrument_type,
            },
            "models": {
                "intensity": self.intensity_model,
                "irt": self.irt_model,
            },
            "prediction_server": self.prediction_server,
            "numThreads": int(self.numThreads),
            "fdr_estimation_method": self.fdr_estimation_method,
            "add_feature_cols": self.add_feature_cols,
            "regressionMethod": self.regressionMethod,
            "ssl": bool(self.ssl),
            "massTolerance": float(self.massTolerance),
            "unitMassTolerance": self.unitMassTolerance,
            "ce_alignment_options": {
                "ce_range": [int(self.ce_range[0]), int(self.ce_range[1])],
                "use_ransac_model": bool(self.use_ransac_model),
            },
        }
        if self.thermoExe:
            cfg["thermoExe"] = self.thermoExe
        if self.static_mods is not None:
            cfg["static_mods"] = self.static_mods
        if self.var_mods is not None:
            cfg["var_mods"] = self.var_mods

        # merge in user overrides at root level (last wins)
        for k, v in (self.extra_config or {}).items():
            cfg[k] = v
        return cfg


def write_oktoberfest_config(cfg: OktoberfestRescoreConfig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return out_path


def run_oktoberfest_rescoring(
    cfg: OktoberfestRescoreConfig,
    *,
    dry_run: bool = False,
    log_path: Optional[Path] = None,
) -> Path:
    """
    Run Oktoberfest rescoring.

    Returns: path to the config file that was used.
    """
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    config_path = cfg.config_path
    if config_path is None:
        config_path = (cfg.work_dir / "oktoberfest_config.json").resolve()
        write_oktoberfest_config(cfg, config_path)

    cmd = [cfg.python_bin, "-m", "oktoberfest", "--config_path", str(config_path)]
    # Use the config file directory as cwd so relative paths behave as expected.
    run_cmd(cmd, cwd=config_path.parent, dry_run=dry_run, log_path=log_path)
    return config_path


def oktoberfest_psm_output_paths(
    *,
    output_dir: Path,
    fdr_method: str,
    kind: str = "rescore",
) -> Tuple[Path, Path]:
    """
    Return (target_psms_path, decoy_psms_path) for a given Oktoberfest run.

    According to the docs, these live under:
      <output_dir>/results/<method>/
        rescore.<method>.psms.txt
        rescore.<method>.decoy.psms.txt

    kind: "rescore" or "original"
    fdr_method: "mokapot" or "percolator"
    """
    method = str(fdr_method).lower()
    sub = output_dir / "results" / method
    tgt = sub / f"{kind}.{method}.psms.txt"
    dec = sub / f"{kind}.{method}.decoy.psms.txt"
    return tgt, dec


def load_oktoberfest_psms(
    psms_path: Path,
    *,
    label: int,
) -> pd.DataFrame:
    """
    Load a single Oktoberfest PSM table (either target or decoy).

    Adds/normalizes:
      - label: 1 target, -1 decoy
      - _q: numeric q-value if present
      - _peptide: best-effort peptide sequence column
      - _proteins: best-effort protein accession/description column
      - _score: best-effort score column (mokapot/percolator)
    """
    df = _read_table_guess(psms_path)

    # q-value columns
    q_col = _find_col(
        df,
        [
            "q-value",
            "q_value",
            "qvalue",
            "qval",
            "mokapot q-value",
            "mokapot q_value",
            "mokapot qvalue",
            "percolator q-value",
            "percolator q_value",
        ],
        required=False,
    )
    if q_col is not None:
        df["_q"] = pd.to_numeric(df[q_col], errors="coerce")
    else:
        df["_q"] = pd.NA

    # Score column (model score)
    score_col = _find_col(
        df,
        [
            "score",
            "mokapot score",
            "mokapot_score",
            "percolator score",
            "percolator_score",
            "svm_score",
        ],
        required=False,
    )
    if score_col is not None:
        df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        df["_score"] = pd.NA

    # peptide sequence
    pep_col = _find_col(
        df,
        [
            "Peptide",
            "peptide",
            "sequence",
            "Sequence",
            "SEQUENCE",
            "peptide_sequence",
            "stripped_peptide",
            "MODIFIED_SEQUENCE",
            "modified_sequence",
        ],
        required=False,
    )
    if pep_col is not None:
        df["_peptide"] = df[pep_col].astype(str)
    else:
        df["_peptide"] = pd.NA

    # proteins (often "Protein" in Oktoberfest tab files; may survive into outputs)
    prot_col = _find_col(
        df,
        [
            "Protein",
            "protein",
            "Proteins",
            "proteins",
            "proteinIds",
            "protein_id",
            "protein_ids",
            "ProteinId",
            "ProteinIds",
        ],
        required=False,
    )
    if prot_col is not None:
        df["_proteins"] = df[prot_col].astype(str)
    else:
        df["_proteins"] = ""

    df["label"] = int(label)
    return df


def load_oktoberfest_results(
    *,
    output_dir: Path,
    fdr_method: str = "mokapot",
    kind: str = "rescore",
    require_files: bool = True,
) -> pd.DataFrame:
    """
    Load Oktoberfest results for analysis.

    Returns a DataFrame with at least:
      - label (1 target, -1 decoy)
      - _q
      - _peptide
      - _proteins
      - _score
    plus all original columns from the underlying output files.

    `kind`:
      - "rescore": features from peptide property prediction
      - "original": only original search-engine features
    """
    tgt_path, dec_path = oktoberfest_psm_output_paths(output_dir=output_dir, fdr_method=fdr_method, kind=kind)

    if require_files:
        if not tgt_path.exists():
            raise FileNotFoundError(f"Missing Oktoberfest target PSM output: {tgt_path}")
        if not dec_path.exists():
            raise FileNotFoundError(f"Missing Oktoberfest decoy PSM output: {dec_path}")

    frames: List[pd.DataFrame] = []
    if tgt_path.exists():
        frames.append(load_oktoberfest_psms(tgt_path, label=1))
    if dec_path.exists():
        frames.append(load_oktoberfest_psms(dec_path, label=-1))
    if not frames:
        raise FileNotFoundError(
            f"Could not find any Oktoberfest PSM outputs under {output_dir}/results/{str(fdr_method).lower()}/"
        )

    out = pd.concat(frames, axis=0, ignore_index=True)
    return out


def oktoberfest_tab_path(
    *,
    output_dir: Path,
    fdr_method: str,
    kind: str = "rescore",
) -> Path:
    """
    Path to the Percolator/Mokapot input tab created by Oktoberfest.

    According to the docs this should exist (for rescoring runs) as:
      <output_dir>/results/<method>/rescore.tab
      <output_dir>/results/<method>/original.tab
    """
    method = str(fdr_method).lower()
    sub = output_dir / "results" / method
    if kind not in ("rescore", "original"):
        raise ValueError("kind must be 'rescore' or 'original'")
    fname = "rescore.tab" if kind == "rescore" else "original.tab"
    return sub / fname


def load_oktoberfest_tab(tab_path: Path) -> pd.DataFrame:
    """
    Load Oktoberfest's percolator/mokapot input tab.

    This file is usually tab-delimited and *should* contain Protein/Label/etc
    (see the Oktoberfest 'Features for target/decoy separation' docs).
    """
    return _read_table_guess(tab_path)


def maybe_attach_proteins_from_tab(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    fdr_method: str = "mokapot",
    kind: str = "rescore",
    min_nonempty_fraction: float = 0.01,
) -> pd.DataFrame:
    """
    If `_proteins` is missing/empty in the loaded Oktoberfest output, try to
    recover it from `rescore.tab` (or `original.tab`) by joining on a shared ID.

    Why this exists:
      In some toolchains, the exported mokapot/percolator output might not carry
      a Protein column through, but the corresponding `*.tab` almost always does.

    Join keys we try (in order):
      SpecId, PSMId, ScanNr, scan, scan_number, SCAN_NUMBER
    """
    if "_proteins" not in df.columns:
        return df
    if df.empty:
        return df

    prot = df["_proteins"].astype(str)
    nonempty = (prot.str.strip() != "").mean()
    if nonempty >= float(min_nonempty_fraction):
        return df  # looks fine

    tab_path = oktoberfest_tab_path(output_dir=output_dir, fdr_method=fdr_method, kind=kind)
    if not tab_path.exists():
        return df

    tab = load_oktoberfest_tab(tab_path)

    join_key = None
    for k in ["SpecId", "PSMId", "ScanNr", "scan", "scan_number", "SCAN_NUMBER"]:
        if k in df.columns and k in tab.columns:
            join_key = k
            break
    if join_key is None:
        return df

    tab_prot_col = _find_col(tab, ["Protein", "Proteins", "protein", "proteins"], required=False)
    if tab_prot_col is None:
        return df

    # Make the mapping 1:1 on join_key to avoid accidental row multiplication.
    tab_map = tab[[join_key, tab_prot_col]].dropna(subset=[join_key]).drop_duplicates(subset=[join_key])

    out = df.merge(tab_map, how="left", on=join_key, suffixes=("", "_from_tab"))
    if "_proteins_from_tab" in out.columns:
        # Fill empty proteins from tab
        m = out["_proteins"].astype(str).str.strip() == ""
        out.loc[m, "_proteins"] = out.loc[m, "_proteins_from_tab"].astype(str)
        out = out.drop(columns=["_proteins_from_tab"])
    return out
