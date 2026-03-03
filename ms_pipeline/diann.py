from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .cmd import run_cmd


def _quote_if_needed(token: str) -> str:
    # DIA-NN config files are tokenized by whitespace; quote tokens with whitespace.
    # DIA-NN accepts quotes in CLI-style (same as GUI “Additional options”).
    if any(ch.isspace() for ch in token):
        # Use double quotes and escape any existing double quotes.
        return '"' + token.replace('"', '\\"') + '"'
    return token


def write_diann_cfg(tokens: Sequence[str], cfg_path: Path) -> Path:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    # One token per line keeps things readable; DIA-NN treats whitespace equivalently.
    with cfg_path.open("w", encoding="utf-8") as fh:
        for t in tokens:
            fh.write(_quote_if_needed(str(t)))
            fh.write("\n")
    return cfg_path


@dataclass
class DiannRunConfig:
    # Binary
    diann_bin: str = "diann"  # e.g. "diann", "/usr/diann/1.8/diann-1.8", "diann.exe"

    # Inputs (choose either raw_files/raw_dir; both allowed)
    raw_files: List[Path] = field(default_factory=list)   # each becomes: --f <file>
    raw_dir: Optional[Path] = None                        # becomes: --dir <folder>

    # FASTA(s) and/or library
    fasta_files: List[Path] = field(default_factory=list)  # each becomes: --fasta <file>
    lib: Optional[Path] = None                             # becomes: --lib <file>

    # Output
    out_report: Path = Path("report.tsv")                  # --out <file>
    out_lib: Optional[Path] = None                         # --out-lib <file>
    temp_dir: Optional[Path] = None                        # --temp <folder>

    # Common options
    threads: Optional[int] = None                          # --threads N
    verbose: Optional[int] = None                          # --verbose N
    qvalue: Optional[float] = None                         # --qvalue X   (optional filter)
    # NOTE: DIA-NN has multiple q-values (run-specific/global; precursor/protein). qvalue here
    # is the CLI filtering threshold if you use it; you may prefer to omit and filter downstream.

    # Library-free mode helpers
    fasta_search: bool = False                             # --fasta-search (in silico digest)
    predictor: bool = False                                # --predictor (DL prediction)
    gen_spec_lib: bool = False                             # --gen-spec-lib (generate spectral library)
    # (You might also want --reanalyse for MBR, etc.)

    # Config file support
    cfg_in: Optional[Path] = None                          # if set: --cfg <cfg_in> is passed
    cfg_out: Optional[Path] = None                         # if set: generate a cfg and pass via --cfg

    # Enzyme / protease
    cut: Optional[str] = None                              # --cut <rule> (e.g. "K*,R*,!*P")
    min_pep_len: Optional[int] = None                      # --min-pep-len N
    max_pep_len: Optional[int] = None                      # --max-pep-len N

    # Additional raw CLI tokens (e.g., ["--missed-cleavages", "2", ...])
    extra_args: List[str] = field(default_factory=list)


def build_diann_cfg_tokens(cfg: DiannRunConfig) -> List[str]:
    tokens: List[str] = []
    if cfg.raw_dir is not None:
        tokens.extend(["--dir", str(cfg.raw_dir)])
    for f in cfg.raw_files:
        tokens.extend(["--f", str(f)])
    return tokens


def build_diann_cmd(cfg: DiannRunConfig) -> Tuple[List[str], Optional[Path]]:
    cmd: List[str] = [cfg.diann_bin]
    used_cfg: Optional[Path] = None

    # If cfg_out is set, generate a cfg that (at least) contains the file list.
    if cfg.cfg_out is not None:
        used_cfg = write_diann_cfg(build_diann_cfg_tokens(cfg), cfg.cfg_out)
        cmd.extend(["--cfg", str(used_cfg)])
    elif cfg.cfg_in is not None:
        used_cfg = cfg.cfg_in
        cmd.extend(["--cfg", str(used_cfg)])
    else:
        # Inline file list
        if cfg.raw_dir is not None:
            cmd.extend(["--dir", str(cfg.raw_dir)])
        for f in cfg.raw_files:
            cmd.extend(["--f", str(f)])

    # FASTA / library
    for fa in cfg.fasta_files:
        cmd.extend(["--fasta", str(fa)])
    if cfg.lib is not None:
        cmd.extend(["--lib", str(cfg.lib)])

    # Mode flags
    if cfg.fasta_search:
        cmd.append("--fasta-search")
    if cfg.predictor:
        cmd.append("--predictor")
    if cfg.gen_spec_lib:
        cmd.append("--gen-spec-lib")

    # Output
    cfg.out_report.parent.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--out", str(cfg.out_report)])
    if cfg.out_lib is not None:
        cfg.out_lib.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--out-lib", str(cfg.out_lib)])
    if cfg.temp_dir is not None:
        cfg.temp_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--temp", str(cfg.temp_dir)])

    # Resources / verbosity
    if cfg.threads is not None and int(cfg.threads) > 0:
        cmd.extend(["--threads", str(int(cfg.threads))])
    if cfg.verbose is not None:
        cmd.extend(["--verbose", str(int(cfg.verbose))])

    # Optional CLI filtering threshold
    if cfg.qvalue is not None:
        cmd.extend(["--qvalue", str(float(cfg.qvalue))])

    # Enzyme / protease
    if cfg.cut is not None:
        cmd.extend(["--cut", cfg.cut])
    if cfg.min_pep_len is not None:
        cmd.extend(["--min-pep-len", str(cfg.min_pep_len)])
    if cfg.max_pep_len is not None:
        cmd.extend(["--max-pep-len", str(cfg.max_pep_len)])

    cmd.append("--report-decoys")
    #cmd.append("--foreign-decoys")

    # Everything else
    cmd.extend([str(x) for x in cfg.extra_args])

    return cmd, used_cfg


def run_diann(
    cfg: DiannRunConfig,
    *,
    dry_run: bool = False,
    log_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    cmd, used_cfg = build_diann_cmd(cfg)
    run_cmd(cmd, cwd=cwd, env=env, dry_run=dry_run, log_path=log_path)

    # On success, DIA-NN writes the report exactly at --out.
    # (Other reports are derived from this name.)
    if dry_run:
        return cfg.out_report
    if cfg.out_report.exists():
        return cfg.out_report
    if cfg.out_report.with_suffix(".parquet").exists():
        return cfg.out_report.with_suffix(".parquet")

    raise FileNotFoundError(f"DIA-NN did not produce expected report: {cfg.out_report}")


def load_diann_report(
    report_path: Path,
    *,
    q_col: Optional[str] = None,
) -> pd.DataFrame:
    # DIA-NN can output .tsv/.txt/.csv; assume tab by default
    if report_path.suffix.lower() == ".tsv":
        df = pd.read_csv(report_path, sep="\t", low_memory=False)
    elif report_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(report_path)
    else:
        raise ValueError(f"Unsupported DIA-NN report format: {report_path.suffix}")

    # q-value column selection
    if q_col is None:
        # Prefer run-specific precursor q-value.
        candidates = [
            "Q.Value",
            "Global.Q.Value",
            "Lib.Q.Value",
            "PG.Q.Value",
            "Global.PG.Q.Value",
            "Lib.PG.Q.Value",
        ]
        q_col = next((c for c in candidates if c in df.columns), None)
        if q_col is None:
            # case-insensitive fallback
            norm = {c.lower().replace("_", "").replace(".", ""): c for c in df.columns}
            for key in ["qvalue", "globalqvalue", "libqvalue"]:
                if key in norm:
                    q_col = norm[key]
                    break
    if q_col is None:
        raise KeyError(f"Could not find a q-value column in DIA-NN report. Columns: {list(df.columns)[:80]} ...")

    df["_q"] = pd.to_numeric(df[q_col], errors="coerce")

    # decoy indicator (best-effort)
    # In many DIA-NN reports, there is a 'Decoy' column (0/1). If absent, label defaults to target.
    decoy_col = None
    for c in ["Decoy", "decoy", "Is.Decoy", "is_decoy", "isDecoy"]:
        if c in df.columns:
            decoy_col = c
            break
    if decoy_col is None:
        df["label"] = 1
    else:
        # interpret 1/True as decoy
        ser = df[decoy_col]
        if ser.dtype == bool:
            is_decoy = ser
        else:
            is_decoy = pd.to_numeric(ser, errors="coerce").fillna(0).astype(int) != 0
        df["label"] = is_decoy.map(lambda x: -1 if bool(x) else 1)

    # peptide column (best-effort; prefer stripped sequence)
    pep_col = None
    for c in [
        "Stripped.Sequence",
        "Stripped.Sequence",
        "Modified.Sequence",
        "Modified.Sequence",
        "Precursor.Id",
        "PrecursorID",
        "Precursor",
        "Sequence",
        "Peptide",
    ]:
        if c in df.columns:
            pep_col = c
            break
    df["_peptide"] = df[pep_col].astype(str) if pep_col is not None else pd.NA

    # proteins column
    prot_col = None
    for c in ["Protein.Ids", "Protein.Group", "Protein.Names", "Protein.IDs", "ProteinIds", "Proteins"]:
        if c in df.columns:
            prot_col = c
            break
    df["_proteins"] = df[prot_col].astype(str) if prot_col is not None else ""

    # score column (CScore is commonly present)
    score_col = None
    for c in ["CScore", "Score", "score", "Mass.Evidence", "PEP"]:
        if c in df.columns:
            score_col = c
            break
    df["_score"] = pd.to_numeric(df[score_col], errors="coerce") if score_col is not None else pd.NA

    return df


def aggregate_to_peptide_level_from_diann(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "_peptide" not in work.columns:
        raise KeyError("Cannot aggregate: missing _peptide column.")
    if "_q" not in work.columns:
        raise KeyError("Cannot aggregate: missing _q column.")

    work["_q"] = pd.to_numeric(work["_q"], errors="coerce")
    work["_score"] = pd.to_numeric(
        work.get("_score", pd.Series([pd.NA] * len(work))), errors="coerce"
    )
    work = work.dropna(subset=["_q"])

    idx = work.groupby("_peptide")["_q"].idxmin()
    cols = ["_peptide", "_q", "label", "_proteins", "_score"]
    cols = [c for c in cols if c in work.columns]
    rep = work.loc[idx, cols].copy()
    rep = rep.rename(columns={"_peptide": "peptide", "_q": "q", "_proteins": "proteins"})
    return rep