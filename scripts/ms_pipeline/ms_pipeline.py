
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from .cmd import run_cmd
from .entrapment import EntrapmentConfig, EntrapmentLabeling, build_combined_fasta
from .sage import load_or_create_sage_config, load_sage_results, aggregate_to_peptide_level_from_sage
from .diann import DiannRunConfig, run_diann, load_diann_report, aggregate_to_peptide_level_from_diann
from .util import sha256_file, plot_ext
from .ms2rescore import load_ms2rescore_results
from .fdr_analysis import analyze_counts_and_entrapment, analyze_counts_and_entrapment_by_length, compute_r_effective, export_by_length_csv
from .plots import plot_counts_vs_q, plot_counts_vs_q_by_length, plot_entrapment_bounds, run_all_length_distribution_plots, run_length_distribution_plots
from .oktoberfest import OktoberfestRescoreConfig, run_oktoberfest_rescoring, load_oktoberfest_results

@dataclass
class ProteaseConfig:
    """Search-engine-agnostic protease description."""
    cleave_at: str                       # Sage enzyme.cleave_at ("" = non-enzymatic)
    restrict: str                        # Sage enzyme.restrict ("" = none)
    c_terminal: bool                     # Sage enzyme.c_terminal
    diann_cut: Optional[str]             # DIA-NN --cut value (None = don't pass)
    min_len: Optional[int] = None        # Override enzyme.min_len (Sage) / --min-pep-len (DIA-NN)
    max_len: Optional[int] = None        # Override enzyme.max_len (Sage) / --max-pep-len (DIA-NN)
    forbid_miscleavages: bool = False


PROTEASE_CONFIGS = {
    "trypsin": ProteaseConfig(cleave_at="KR", restrict="P", c_terminal=True, diann_cut="K*,R*,!*P",),
    "chymotrypsin": ProteaseConfig(cleave_at="FWY", restrict="P", c_terminal=True, diann_cut="F*,W*,Y*,!*P"),
    "pepsin": ProteaseConfig(cleave_at="FWYL", restrict="", c_terminal=True, diann_cut="F*,W*,Y*,L*"),
    "aspn": ProteaseConfig(cleave_at="D", restrict="", c_terminal=False, diann_cut="*D"),
    "gluc": ProteaseConfig(cleave_at="E", restrict="", c_terminal=True, diann_cut="E*"),
    "lysc": ProteaseConfig(cleave_at="K", restrict="", c_terminal=True, diann_cut="K*"),
    "lysn": ProteaseConfig(cleave_at="K", restrict="", c_terminal=False, diann_cut="*K"),
    "argc": ProteaseConfig(cleave_at="R", restrict="", c_terminal=True, diann_cut="R*"),
    "hla": ProteaseConfig(cleave_at="", restrict="", c_terminal=True, diann_cut=None, min_len=7, max_len=11, forbid_miscleavages=True),
}


def _coalesce_cols(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of candidate columns found: {candidates}. Columns: {list(df.columns)[:50]} ...")
    return candidates[0]



@dataclass
class PipelineInputs:
    spectra: List[Path]
    target_fasta: Path
    output_dir: Path
    entrap_cfg: EntrapmentConfig
    search_engine: str = "sage"  # "sage" or "diann"
    fragmentation: str = "hcd"  # "hcd" or "etd"
    protease: str = "trypsin"
    protease_cfg: ProteaseConfig = None  # resolved config (set in main)
    sage_bin: str = "sage"
    experiment_name: str = ""
    sage_config_in: Optional[Path] = None
    sample_psms: Optional[int] = None
    ms2rescore_bin: str = "ms2rescore"
    ms2rescore_config: Optional[Path] = None
    run_ms2rescore: bool = False
    ms2rescore_processes: int = 0  # 0 means "don't pass"
    run_oktoberfest: bool = False
    spectrum_path_for_ms2rescore: Optional[Path] = None
    dry_run: bool = False
    # DIA-NN options
    diann_bin: str = "diann"
    diann_lib: Optional[Path] = None
    diann_fasta_search: bool = False
    diann_predictor: bool = False
    diann_gen_spec_lib: bool = False
    diann_threads: Optional[int] = None
    diann_qvalue: Optional[float] = None
    diann_extra_args: Optional[List[str]] = None


def common_parent(paths: List[Path]) -> Path:
    if not paths:
        return Path(".")
    parts = [p.resolve().parts for p in paths]
    min_len = min(len(x) for x in parts)
    common = []
    for i in range(min_len):
        token = parts[0][i]
        if all(x[i] == token for x in parts):
            common.append(token)
        else:
            break
    return Path(*common) if common else Path(".")


def run_pipeline(inputs: PipelineInputs) -> None:
    out = inputs.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "pipeline.log"

    # 1) Build combined FASTA
    fasta_dir = out / "fasta"
    combined_fasta = fasta_dir / "search.fasta"
    combined_fasta, ent_meta = build_combined_fasta(
        inputs.target_fasta,
        entrap_cfg=inputs.entrap_cfg,
        out_fasta=combined_fasta,
        work_dir=out,
    )

    # 2) Branch on search engine
    experiment_name = inputs.experiment_name
    sage_results_tsv = None
    diann_report = None
    ms2rescore_psms_tsv = None
    odf = None  # oktoberfest dataframe, if used

    if inputs.search_engine == "sage":
        experiment_name += "_sage"
        if inputs.run_ms2rescore:
            experiment_name += "_ms2rescore"
        if inputs.run_oktoberfest:
            experiment_name += "_oktoberfest"

        # Sage config snapshot
        cfg_dir = out / "configs"
        sage_cfg_path = cfg_dir / "sage_config.json"
        load_or_create_sage_config(
            sage_config_in=inputs.sage_config_in,
            out_path=sage_cfg_path,
            entrap_cfg=inputs.entrap_cfg,
            fragmentation=inputs.fragmentation,
            protease_cfg=inputs.protease_cfg,
        )

        # Run Sage
        sage_out = out / "sage"
        sage_out.mkdir(parents=True, exist_ok=True)

        sage_cmd = [
            inputs.sage_bin,
            str(sage_cfg_path),
            "-o",
            str(sage_out),
            "-f",
            str(combined_fasta),
            "--write-pin",
        ] + [str(p) for p in inputs.spectra]

        run_cmd(sage_cmd, dry_run=inputs.dry_run, log_path=log_path)

        sage_results_tsv = sage_out / "results.sage.tsv"

        if inputs.sample_psms is not None and not inputs.dry_run:
            sage_df = pd.read_csv(sage_results_tsv, sep="\t")
            sage_df = sage_df.sample(n=inputs.sample_psms, random_state=42)

            sampled_path = sage_out / f"results.sage.sample_{inputs.sample_psms}.tsv"
            sage_df.to_csv(sampled_path, sep="\t", index=False)
            sage_results_tsv = sampled_path

        # Optional MS2Rescore
        ms2rescore_out_prefix = None
        if inputs.run_ms2rescore:
            rescore_dir = out / "ms2rescore"
            rescore_dir.mkdir(parents=True, exist_ok=True)
            ms2rescore_out_prefix = rescore_dir / "rescore"
            spectrum_path = inputs.spectrum_path_for_ms2rescore
            if spectrum_path is None:
                spectrum_path = common_parent(inputs.spectra).resolve()
                print(spectrum_path)
            ms2_cmd = [
                inputs.ms2rescore_bin,
                "-p",
                str(sage_results_tsv),
                "-t",
                "sage",
                "-s",
                str(spectrum_path),
                "-f",
                str(combined_fasta),
                "-o",
                str(ms2rescore_out_prefix),
            ]
            if inputs.ms2rescore_config is not None:
                ms2_cmd.extend(["-c", str(inputs.ms2rescore_config)])
            if inputs.ms2rescore_processes and inputs.ms2rescore_processes > 0:
                ms2_cmd.extend(["-n", str(inputs.ms2rescore_processes)])

            run_cmd(ms2_cmd, dry_run=inputs.dry_run, log_path=log_path)
            ms2rescore_psms_tsv = Path(str(ms2rescore_out_prefix) + ".psms.tsv")
            if not inputs.dry_run and not ms2rescore_psms_tsv.exists():
                raise FileNotFoundError(f"Expected MS2Rescore output not found: {ms2rescore_psms_tsv}")

        # Optional Oktoberfest
        if inputs.run_oktoberfest:
            oktoberfest_out_prefix = out / "oktoberfest"
            oktoberfest_out_prefix.mkdir(parents=True, exist_ok=True)

            spectra_path = common_parent(inputs.spectra).resolve()
            okt_cfg = OktoberfestRescoreConfig(
                work_dir=oktoberfest_out_prefix,
                search_results=sage_results_tsv.resolve(),
                spectra=spectra_path,
                search_results_type="Sage",
                spectra_type="mzml",
                instrument_type="QE",
                fdr_estimation_method="percolator",
            )

            run_oktoberfest_rescoring(okt_cfg, dry_run=inputs.dry_run, log_path=log_path)

            okt_output_dir = oktoberfest_out_prefix
            odf = load_oktoberfest_results(
                output_dir=okt_output_dir,
                fdr_method=okt_cfg.fdr_estimation_method,
                kind="rescore",
            )
            odf = odf.rename(columns={"_q": "q", "_proteins": "proteins"})

    elif inputs.search_engine == "diann":
        experiment_name += "_diann"

        # Run DIA-NN
        diann_out = out / "diann"
        diann_out.mkdir(parents=True, exist_ok=True)

        diann_cfg = DiannRunConfig(
            diann_bin=inputs.diann_bin,
            raw_files=inputs.spectra,
            fasta_files=[combined_fasta],
            lib=inputs.diann_lib,
            out_report=diann_out / "report.tsv",
            temp_dir=diann_out / "temp",
            threads=inputs.diann_threads,
            fasta_search=inputs.diann_fasta_search,
            predictor=inputs.diann_predictor,
            gen_spec_lib=inputs.diann_gen_spec_lib,
            qvalue=inputs.diann_qvalue,
            cut=inputs.protease_cfg.diann_cut,
            min_pep_len=inputs.protease_cfg.min_len,
            max_pep_len=inputs.protease_cfg.max_len,
            extra_args=inputs.diann_extra_args or [],
        )

        diann_report = run_diann(diann_cfg, dry_run=inputs.dry_run, log_path=log_path)

    else:
        raise ValueError(f"Unknown search engine: {inputs.search_engine}")

    # Write run metadata
    run_meta = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "search_engine": inputs.search_engine,
        "fragmentation": inputs.fragmentation,
        "protease": inputs.protease,
        "spectra": [str(p) for p in inputs.spectra],
        "target_fasta": str(inputs.target_fasta),
        "combined_fasta": str(combined_fasta),
        "entrapment": ent_meta,
        "file_hashes": {},
    }
    if inputs.search_engine == "sage":
        run_meta["sage_config"] = str(sage_cfg_path)
        run_meta["sage_results"] = str(sage_results_tsv)
        run_meta["run_ms2rescore"] = inputs.run_ms2rescore
        run_meta["ms2rescore_output_prefix"] = str(ms2rescore_out_prefix) if ms2rescore_out_prefix else None
        run_meta["ms2rescore_psms_tsv"] = str(ms2rescore_psms_tsv) if ms2rescore_psms_tsv else None
    elif inputs.search_engine == "diann":
        run_meta["diann_report"] = str(diann_report)
    # hashes for reproducibility (skip spectra, too large)
    try:
        run_meta["file_hashes"]["target_fasta_sha256"] = sha256_file(inputs.target_fasta)
        run_meta["file_hashes"]["combined_fasta_sha256"] = sha256_file(combined_fasta)
        if inputs.search_engine == "sage":
            run_meta["file_hashes"]["sage_config_sha256"] = sha256_file(sage_cfg_path)
    except Exception:
        pass

    (out / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    # Analysis
    if inputs.dry_run:
        print("Dry run enabled; skipping analysis.")
        return

    analysis_dir = out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    labeling = EntrapmentLabeling(
        entrapment_prefix=inputs.entrap_cfg.entrapment_prefix,
        entrapment_strategy=inputs.entrap_cfg.entrapment_strategy,
        pairing_regex=inputs.entrap_cfg.pairing_regex,
    )

    # Compute r if foreign entrapment
    r_effective = None
    _pcfg = inputs.protease_cfg
    if inputs.entrap_cfg.mode == "foreign" and inputs.entrap_cfg.entrapment_fasta is not None:
        try:
            r_effective = compute_r_effective(
                target_fasta=inputs.target_fasta,
                entrapment_fasta=inputs.entrap_cfg.entrapment_fasta,
                enzyme_cleave_at=_pcfg.cleave_at,
                restrict=_pcfg.restrict,
                missed_cleavages=2,
                min_len=_pcfg.min_len or 7,
                max_len=_pcfg.max_len or 50,
                c_terminal=_pcfg.c_terminal,
                collapse_il=inputs.entrap_cfg.collapse_il,
            )
        except Exception as e:
            print(f"Warning: failed to compute r_effective from FASTA digestion: {e}")
            r_effective = None

    # ================================================================
    # Sage analysis
    # ================================================================
    if inputs.search_engine == "sage":
        # Try to refine r_effective from Sage config if foreign entrapment
        if inputs.entrap_cfg.mode == "foreign" and inputs.entrap_cfg.entrapment_fasta is not None:
            try:
                cfg = json.loads(sage_cfg_path.read_text(encoding="utf-8"))
                enz = cfg.get("database", {}).get("enzyme", {})
                r_effective = compute_r_effective(
                    target_fasta=inputs.target_fasta,
                    entrapment_fasta=inputs.entrap_cfg.entrapment_fasta,
                    enzyme_cleave_at=str(enz.get("cleave_at", "KR")),
                    restrict=str(enz.get("restrict", "P")),
                    missed_cleavages=int(enz.get("missed_cleavages", 2)),
                    min_len=int(enz.get("min_len", 7)),
                    max_len=int(enz.get("max_len", 50)),
                    c_terminal=bool(enz.get("c_terminal", True)),
                    collapse_il=inputs.entrap_cfg.collapse_il,
                )
            except Exception as e:
                print(f"Warning: failed to compute r_effective from Sage config: {e}")

        # ---- Sage PSM-level analysis
        sage_df = load_sage_results(sage_results_tsv)
        proteins_col = "proteins" if "proteins" in sage_df.columns else _coalesce_cols(sage_df, ["protein", "_proteins"], required=True)

        q_col = "spectrum_q" if "spectrum_q" in sage_df.columns else _coalesce_cols(sage_df, ["q", "_q"], required=True)
        res = analyze_counts_and_entrapment(
            sage_df,
            q_col=q_col,
            proteins_col=proteins_col,
            labeling=labeling,
            r_effective=r_effective,
        )
        res["counts_vs_q"].to_csv(analysis_dir / "sage_psm_counts_vs_q.csv", index=False)
        res["entrapment_bounds_vs_q"].to_csv(analysis_dir / "sage_psm_entrapment_bounds_vs_q.csv", index=False)
        plot_counts_vs_q(res["counts_vs_q"], analysis_dir / f"sage_psm_counts_vs_q{plot_ext()}", title=f"Sage PSM counts vs q ({experiment_name})")
        plot_entrapment_bounds(res["entrapment_bounds_vs_q"], analysis_dir / f"sage_psm_entrapment_bounds{plot_ext()}", title=f"Sage entrapment FDP bounds (PSM-level) ({experiment_name})")

        # Score distribution plots - Sage PSM level
        _pep_col_sage = "stripped_peptide" if "stripped_peptide" in sage_df.columns else "peptide"

        try:
            by_len = analyze_counts_and_entrapment_by_length(
                sage_df, q_col=q_col, proteins_col=proteins_col, peptide_col=_pep_col_sage,
                labeling=labeling, r_effective=r_effective,
            )
            if by_len:
                plot_counts_vs_q_by_length(by_len, analysis_dir / f"sage_psm_counts_vs_q_by_length{plot_ext()}", title=f"Sage PSM counts vs q by length ({experiment_name})")
                export_by_length_csv(by_len, analysis_dir / "sage_psm_counts_vs_q_by_length.csv")
        except Exception as e:
            print(f"Warning: Sage PSM per-length counts_vs_q failed: {e}")
        try:
            run_all_length_distribution_plots(
                sage_df,
                proteins_col=proteins_col,
                label_col="label",
                entrapment_prefix=inputs.entrap_cfg.entrapment_prefix,
                peptide_col=_pep_col_sage,
                out_dir=analysis_dir,
                file_prefix="sage_psm",
                title_prefix="Sage PSM-level",
                experiment_name=experiment_name,
            )
        except Exception as e:
            print(f"Warning: Sage PSM score distribution plots failed: {e}")

        # ---- Sage peptide-level (via peptide_q)
        try:
            pep_df = aggregate_to_peptide_level_from_sage(sage_df)
            pep_df = pep_df.rename(columns={"q": "peptide_q"})
            pep_res = analyze_counts_and_entrapment(
                pep_df.rename(columns={"peptide_q": "q"}),
                q_col="q",
                proteins_col="proteins",
                labeling=labeling,
                r_effective=r_effective,
            )
            pep_res["counts_vs_q"].to_csv(analysis_dir / "sage_peptide_counts_vs_q.csv", index=False)
            pep_res["entrapment_bounds_vs_q"].to_csv(analysis_dir / "sage_peptide_entrapment_bounds_vs_q.csv", index=False)
            plot_counts_vs_q(pep_res["counts_vs_q"], analysis_dir / f"sage_peptide_counts_vs_q{plot_ext()}", title=f"Sage peptide counts vs q ({experiment_name})")
            plot_entrapment_bounds(pep_res["entrapment_bounds_vs_q"], analysis_dir / f"sage_peptide_entrapment_bounds{plot_ext()}", title=f"Sage entrapment FDP bounds (peptide-level) ({experiment_name})")

            try:
                pep_by_len = analyze_counts_and_entrapment_by_length(
                    pep_df.rename(columns={"peptide_q": "q"}), q_col="q", proteins_col="proteins", peptide_col="peptide",
                    labeling=labeling, r_effective=r_effective,
                )
                if pep_by_len:
                    plot_counts_vs_q_by_length(pep_by_len, analysis_dir / f"sage_peptide_counts_vs_q_by_length{plot_ext()}", title=f"Sage peptide counts vs q by length ({experiment_name})")
                    export_by_length_csv(pep_by_len, analysis_dir / "sage_peptide_counts_vs_q_by_length.csv")
            except Exception as e:
                print(f"Warning: Sage peptide per-length counts_vs_q failed: {e}")

            # Score distribution plots - Sage peptide level
            print(pep_df.columns)
            run_all_length_distribution_plots(
                pep_df,
                proteins_col="proteins",
                label_col="label",
                entrapment_prefix=inputs.entrap_cfg.entrapment_prefix,
                peptide_col="peptide",
                out_dir=analysis_dir,
                file_prefix="sage_peptide",
                title_prefix="Sage Peptide-level",
                experiment_name=experiment_name,
            )
        except Exception as e:
            print(f"Warning: peptide-level aggregation/analysis failed: {e}")

        # ---- MS2Rescore analysis (if present)
        if ms2rescore_psms_tsv is not None:
            try:
                mdf = load_ms2rescore_results(ms2rescore_psms_tsv)
                mdf = mdf.rename(columns={"_q": "q", "_proteins": "proteins"})
                ores = analyze_counts_and_entrapment(
                    mdf,
                    q_col="q",
                    proteins_col="proteins",
                    labeling=labeling,
                    r_effective=r_effective,
                )
                ores["counts_vs_q"].to_csv(analysis_dir / "ms2rescore_psm_counts_vs_q.csv", index=False)
                ores["entrapment_bounds_vs_q"].to_csv(analysis_dir / "ms2rescore_psm_entrapment_bounds_vs_q.csv", index=False)
                plot_counts_vs_q(ores["counts_vs_q"], analysis_dir / f"ms2rescore_psm_counts_vs_q{plot_ext()}", title=f"MS2Rescore PSM counts vs q ({experiment_name})")
                plot_entrapment_bounds(ores["entrapment_bounds_vs_q"], analysis_dir / f"ms2rescore_psm_entrapment_bounds{plot_ext()}", title=f"MS2Rescore entrapment FDP bounds (PSM-level) ({experiment_name})")

                ms2_pep_col = "_peptide" if "_peptide" in mdf.columns else "peptide"
                try:
                    ms2_by_len = analyze_counts_and_entrapment_by_length(
                        mdf, q_col="q", proteins_col="proteins", peptide_col=ms2_pep_col,
                        labeling=labeling, r_effective=r_effective,
                    )
                    if ms2_by_len:
                        plot_counts_vs_q_by_length(ms2_by_len, analysis_dir / f"ms2rescore_psm_counts_vs_q_by_length{plot_ext()}", title=f"MS2Rescore PSM counts vs q by length ({experiment_name})")
                        export_by_length_csv(ms2_by_len, analysis_dir / "ms2rescore_psm_counts_vs_q_by_length.csv")
                except Exception as e:
                    print(f"Warning: MS2Rescore PSM per-length counts_vs_q failed: {e}")
                run_all_length_distribution_plots(
                    mdf,
                    proteins_col="proteins",
                    label_col="label",
                    entrapment_prefix=inputs.entrap_cfg.entrapment_prefix,
                    peptide_col=ms2_pep_col,
                    out_dir=analysis_dir,
                    file_prefix="ms2rescore_psm",
                    title_prefix="MS2Rescore PSM-level",
                    experiment_name=experiment_name,
                )

                # MS2Rescore peptide-level
                try:
                    ms2_pep_df = mdf.dropna(subset=[ms2_pep_col]).copy()
                    ms2_pep_df["q"] = pd.to_numeric(ms2_pep_df["q"], errors="coerce")
                    ms2_pep_df = ms2_pep_df.dropna(subset=["q"])
                    idx = ms2_pep_df.groupby(ms2_pep_col)["q"].idxmin()
                    ms2_pep_agg = ms2_pep_df.loc[idx, [ms2_pep_col, "q", "label", "proteins", "_score"]].copy()
                    ms2_pep_agg = ms2_pep_agg.rename(columns={ms2_pep_col: "peptide"})

                    ms2_pep_res = analyze_counts_and_entrapment(
                        ms2_pep_agg, q_col="q", proteins_col="proteins",
                        labeling=labeling, r_effective=r_effective,
                    )
                    ms2_pep_res["counts_vs_q"].to_csv(analysis_dir / "ms2rescore_peptide_counts_vs_q.csv", index=False)
                    ms2_pep_res["entrapment_bounds_vs_q"].to_csv(analysis_dir / "ms2rescore_peptide_entrapment_bounds_vs_q.csv", index=False)
                    plot_counts_vs_q(ms2_pep_res["counts_vs_q"], analysis_dir / f"ms2rescore_peptide_counts_vs_q{plot_ext()}", title=f"MS2Rescore peptide counts vs q ({experiment_name})")
                    plot_entrapment_bounds(ms2_pep_res["entrapment_bounds_vs_q"], analysis_dir / f"ms2rescore_peptide_entrapment_bounds{plot_ext()}", title=f"MS2Rescore entrapment FDP bounds (peptide-level) ({experiment_name})")

                    try:
                        ms2_pep_by_len = analyze_counts_and_entrapment_by_length(
                            ms2_pep_agg, q_col="q", proteins_col="proteins", peptide_col="peptide",
                            labeling=labeling, r_effective=r_effective,
                        )
                        if ms2_pep_by_len:
                            plot_counts_vs_q_by_length(ms2_pep_by_len, analysis_dir / f"ms2rescore_peptide_counts_vs_q_by_length{plot_ext()}", title=f"MS2Rescore peptide counts vs q by length ({experiment_name})")
                            export_by_length_csv(ms2_pep_by_len, analysis_dir / "ms2rescore_peptide_counts_vs_q_by_length.csv")
                    except Exception as e:
                        print(f"Warning: MS2Rescore peptide per-length counts_vs_q failed: {e}")

                    run_all_length_distribution_plots(
                        ms2_pep_agg,
                        proteins_col="proteins", label_col="label",
                        entrapment_prefix=inputs.entrap_cfg.entrapment_prefix,
                        peptide_col="peptide", out_dir=analysis_dir,
                        file_prefix="ms2rescore_peptide", title_prefix="MS2Rescore Peptide-level",
                        experiment_name=experiment_name,
                    )
                except Exception as e:
                    print(f"Warning: MS2Rescore peptide-level analysis failed: {e}")
            except Exception as e:
                print(f"Warning: MS2Rescore analysis failed: {e}")

        # ---- Oktoberfest analysis (if present)
        if inputs.run_oktoberfest and odf is not None:
            try:
                ores = analyze_counts_and_entrapment(
                    odf, q_col="q", proteins_col="proteins",
                    labeling=labeling, r_effective=r_effective,
                )
                ores["counts_vs_q"].to_csv(analysis_dir / "oktoberfest_psm_counts_vs_q.csv", index=False)
                ores["entrapment_bounds_vs_q"].to_csv(analysis_dir / "oktoberfest_psm_entrapment_bounds_vs_q.csv", index=False)
                plot_counts_vs_q(ores["counts_vs_q"], analysis_dir / f"oktoberfest_psm_counts_vs_q{plot_ext()}", title=f"Oktoberfest PSM counts vs q ({experiment_name})")
                plot_entrapment_bounds(ores["entrapment_bounds_vs_q"], analysis_dir / f"oktoberfest_psm_entrapment_bounds{plot_ext()}", title=f"Oktoberfest entrapment FDP bounds (PSM-level) ({experiment_name})")

                ms2_pep_col = "_peptide" if "_peptide" in odf.columns else "peptide"
                try:
                    okt_by_len = analyze_counts_and_entrapment_by_length(
                        odf, q_col="q", proteins_col="proteins", peptide_col=ms2_pep_col,
                        labeling=labeling, r_effective=r_effective,
                    )
                    if okt_by_len:
                        plot_counts_vs_q_by_length(okt_by_len, analysis_dir / f"oktoberfest_psm_counts_vs_q_by_length{plot_ext()}", title=f"Oktoberfest PSM counts vs q by length ({experiment_name})")
                        export_by_length_csv(okt_by_len, analysis_dir / "oktoberfest_psm_counts_vs_q_by_length.csv")
                except Exception as e:
                    print(f"Warning: Oktoberfest PSM per-length counts_vs_q failed: {e}")
                run_all_length_distribution_plots(
                    odf,
                    proteins_col="proteins", label_col="label",
                    entrapment_prefix=inputs.entrap_cfg.entrapment_prefix,
                    peptide_col=ms2_pep_col, out_dir=analysis_dir,
                    file_prefix="oktoberfest_psm", title_prefix="Oktoberfest PSM-level",
                    experiment_name=experiment_name,
                )

                # Oktoberfest peptide-level
                try:
                    okt_pep_df = odf.dropna(subset=[ms2_pep_col]).copy()
                    okt_pep_df["q"] = pd.to_numeric(okt_pep_df["q"], errors="coerce")
                    okt_pep_df = okt_pep_df.dropna(subset=["q"])
                    idx = okt_pep_df.groupby(ms2_pep_col)["q"].idxmin()
                    okt_pep_agg = okt_pep_df.loc[idx, [ms2_pep_col, "q", "label", "proteins", "_score"]].copy()
                    okt_pep_agg = okt_pep_agg.rename(columns={ms2_pep_col: "peptide"})

                    okt_pep_res = analyze_counts_and_entrapment(
                        okt_pep_agg, q_col="q", proteins_col="proteins",
                        labeling=labeling, r_effective=r_effective,
                    )
                    okt_pep_res["counts_vs_q"].to_csv(analysis_dir / "oktoberfest_peptide_counts_vs_q.csv", index=False)
                    okt_pep_res["entrapment_bounds_vs_q"].to_csv(analysis_dir / "oktoberfest_peptide_entrapment_bounds_vs_q.csv", index=False)
                    plot_counts_vs_q(okt_pep_res["counts_vs_q"], analysis_dir / f"oktoberfest_peptide_counts_vs_q{plot_ext()}", title=f"Oktoberfest peptide counts vs q ({experiment_name})")
                    plot_entrapment_bounds(okt_pep_res["entrapment_bounds_vs_q"], analysis_dir / f"oktoberfest_peptide_entrapment_bounds{plot_ext()}", title=f"Oktoberfest entrapment FDP bounds (peptide-level) ({experiment_name})")

                    try:
                        okt_pep_by_len = analyze_counts_and_entrapment_by_length(
                            okt_pep_agg, q_col="q", proteins_col="proteins", peptide_col="peptide",
                            labeling=labeling, r_effective=r_effective,
                        )
                        if okt_pep_by_len:
                            plot_counts_vs_q_by_length(okt_pep_by_len, analysis_dir / f"oktoberfest_peptide_counts_vs_q_by_length{plot_ext()}", title=f"Oktoberfest peptide counts vs q by length ({experiment_name})")
                            export_by_length_csv(okt_pep_by_len, analysis_dir / "oktoberfest_peptide_counts_vs_q_by_length.csv")
                    except Exception as e:
                        print(f"Warning: Oktoberfest peptide per-length counts_vs_q failed: {e}")

                    run_all_length_distribution_plots(
                        okt_pep_agg,
                        proteins_col="proteins", label_col="label",
                        entrapment_prefix=inputs.entrap_cfg.entrapment_prefix,
                        peptide_col="peptide", out_dir=analysis_dir,
                        file_prefix="oktoberfest_peptide", title_prefix="Oktoberfest Peptide-level",
                        experiment_name=experiment_name,
                    )
                except Exception as e:
                    print(f"Warning: Oktoberfest peptide-level analysis failed: {e}")
            except Exception as e:
                print(f"Warning: Oktoberfest analysis failed: {e}")

    # ================================================================
    # DIA-NN analysis
    # ================================================================
    elif inputs.search_engine == "diann":
        diann_df = load_diann_report(diann_report)
        # Normalize column names for analysis functions
        diann_df = diann_df.rename(columns={"_proteins": "proteins"})
        proteins_col = "proteins"

        # ---- DIA-NN precursor-level analysis
        q_col = "_q"
        res = analyze_counts_and_entrapment(
            diann_df,
            q_col=q_col,
            proteins_col=proteins_col,
            labeling=labeling,
            r_effective=r_effective,
        )
        res["counts_vs_q"].to_csv(analysis_dir / "diann_precursor_counts_vs_q.csv", index=False)
        res["entrapment_bounds_vs_q"].to_csv(analysis_dir / "diann_precursor_entrapment_bounds_vs_q.csv", index=False)
        plot_counts_vs_q(res["counts_vs_q"], analysis_dir / f"diann_precursor_counts_vs_q{plot_ext()}", title=f"DIA-NN precursor counts vs q ({experiment_name})")
        plot_entrapment_bounds(res["entrapment_bounds_vs_q"], analysis_dir / f"diann_precursor_entrapment_bounds{plot_ext()}", title=f"DIA-NN entrapment FDP bounds (precursor-level) ({experiment_name})")

        # Score distribution plots - DIA-NN precursor level
        pep_col_diann = "_peptide"
        try:
            diann_by_len = analyze_counts_and_entrapment_by_length(
                diann_df, q_col=q_col, proteins_col=proteins_col, peptide_col=pep_col_diann,
                labeling=labeling, r_effective=r_effective,
            )
            if diann_by_len:
                plot_counts_vs_q_by_length(diann_by_len, analysis_dir / f"diann_precursor_counts_vs_q_by_length{plot_ext()}", title=f"DIA-NN precursor counts vs q by length ({experiment_name})")
                export_by_length_csv(diann_by_len, analysis_dir / "diann_precursor_counts_vs_q_by_length.csv")
        except Exception as e:
            print(f"Warning: DIA-NN precursor per-length counts_vs_q failed: {e}")
        try:
            run_all_length_distribution_plots(
                diann_df,
                proteins_col=proteins_col,
                label_col="label",
                entrapment_prefix=inputs.entrap_cfg.entrapment_prefix,
                peptide_col=pep_col_diann,
                out_dir=analysis_dir,
                file_prefix="diann_precursor",
                title_prefix="DIA-NN Precursor-level",
                experiment_name=experiment_name,
            )
        except Exception as e:
            print(f"Warning: DIA-NN precursor score distribution plots failed: {e}")

        # ---- DIA-NN peptide-level
        try:
            pep_df = aggregate_to_peptide_level_from_diann(diann_df)
            pep_res = analyze_counts_and_entrapment(
                pep_df,
                q_col="q",
                proteins_col="proteins",
                labeling=labeling,
                r_effective=r_effective,
            )
            pep_res["counts_vs_q"].to_csv(analysis_dir / "diann_peptide_counts_vs_q.csv", index=False)
            pep_res["entrapment_bounds_vs_q"].to_csv(analysis_dir / "diann_peptide_entrapment_bounds_vs_q.csv", index=False)
            plot_counts_vs_q(pep_res["counts_vs_q"], analysis_dir / f"diann_peptide_counts_vs_q{plot_ext()}", title=f"DIA-NN peptide counts vs q ({experiment_name})")
            plot_entrapment_bounds(pep_res["entrapment_bounds_vs_q"], analysis_dir / f"diann_peptide_entrapment_bounds{plot_ext()}", title=f"DIA-NN entrapment FDP bounds (peptide-level) ({experiment_name})")

            try:
                diann_pep_by_len = analyze_counts_and_entrapment_by_length(
                    pep_df, q_col="q", proteins_col="proteins", peptide_col="peptide",
                    labeling=labeling, r_effective=r_effective,
                )
                if diann_pep_by_len:
                    plot_counts_vs_q_by_length(diann_pep_by_len, analysis_dir / f"diann_peptide_counts_vs_q_by_length{plot_ext()}", title=f"DIA-NN peptide counts vs q by length ({experiment_name})")
                    export_by_length_csv(diann_pep_by_len, analysis_dir / "diann_peptide_counts_vs_q_by_length.csv")
            except Exception as e:
                print(f"Warning: DIA-NN peptide per-length counts_vs_q failed: {e}")

            run_all_length_distribution_plots(
                pep_df,
                proteins_col="proteins",
                label_col="label",
                entrapment_prefix=inputs.entrap_cfg.entrapment_prefix,
                peptide_col="peptide",
                out_dir=analysis_dir,
                file_prefix="diann_peptide",
                title_prefix="DIA-NN Peptide-level",
                experiment_name=experiment_name,
            )
        except Exception as e:
            print(f"Warning: DIA-NN peptide-level analysis failed: {e}")


    print(f"Done. Outputs written to: {out}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Sage or DIA-NN (with optional rescoring) and compute Wen et al.-style entrapment validation curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spectra", nargs="+", required=True, help="One or more mzML/mgf/raw files to search.")
    p.add_argument("--target-fasta", required=True, help="Target FASTA (e.g., human).")
    p.add_argument("--output-dir", required=True, help="Output directory for all run products.")

    p.add_argument("--search-engine", default="sage", choices=["sage", "diann"],
                    help="Search engine to use. MS2Rescore and Oktoberfest rescoring are only available with Sage.")
    p.add_argument("--fragmentation", default="hcd", type=str.lower, choices=["hcd", "etd"],
                    help="Fragmentation type. ETD uses c/z ions instead of b/y. "
                         "ETD is only supported with Sage (not with DIA-NN, MS2Rescore, or Oktoberfest).")
    p.add_argument("--protease", default="trypsin", type=str.lower, choices=list(PROTEASE_CONFIGS),
                    help="Protease used for digestion. Configures enzyme in Sage and --cut in DIA-NN.")
    p.add_argument("--hla-min-len", type=int, default=7, help="Min peptide length for HLA non-specific digestion.")
    p.add_argument("--hla-max-len", type=int, default=11, help="Max peptide length for HLA non-specific digestion.")
    p.add_argument("--decoy-prefix", default="rev_", help="Prefix of decoys in FASTA files provided. If empty, decoys are generated by Sage.")

    # Sage options
    p.add_argument("--sage-bin", default="sage", help="Sage executable (default: sage).")
    p.add_argument("--sage-config", default=None, help="Base Sage JSON config. If omitted, a sensible default is used.")

    p.add_argument("--experiment-name", default="", help="Name for this experiment/run, used in plot titles. (default: empty)")

    # Entrapment options
    p.add_argument(
        "--entrapment-mode",
        default="none",
        choices=["none", "foreign", "paired_shuffled_peptides"],
        help="Entrapment strategy: none | foreign FASTA | paired shuffled peptide DB.",
    )
    p.add_argument("--entrapment-fasta", default=None, help="Entrapment FASTA (required for entrapment-mode=foreign).")
    p.add_argument("--entrapment-prefix", default="ENTRAP_", help="Prefix added to entrapment accessions in combined FASTA.")
    p.add_argument("--entrapment-strategy", default="unambiguous", choices=["unambiguous", "any"], help="How to classify hits as entrapment.")
    p.add_argument("--collapse-il", action="store_true", help="For peptide DB generation + r estimation: treat I as L.")

    # Paired mode options
    p.add_argument("--paired-seed", type=int, default=1, help="Random seed for paired shuffled entrapments.")
    p.add_argument("--paired-min-len", type=int, default=7, help="Min peptide length for paired peptide DB.")
    p.add_argument("--paired-max-len", type=int, default=35, help="Max peptide length for paired peptide DB.")
    p.add_argument("--paired-missed-cleavages", type=int, default=2, help="Missed cleavages when digesting proteins to peptides for paired mode.")
    p.add_argument(
        "--paired-keep-c-terminal",
        dest="paired_keep_c_terminal",
        action="store_true",
        default=True,
        help="Keep C-terminal AA fixed when shuffling (tryptic-friendly). (default: enabled)",
    )
    p.add_argument(
        "--paired-no-keep-c-terminal",
        dest="paired_keep_c_terminal",
        action="store_false",
        help="Disable keeping the C-terminal AA fixed when shuffling.",
    )
    p.add_argument("--pairing-regex", default=r"^(TGT|ENTRAP)\|(\d+)$", help="Regex to parse (kind,pair_id) for paired estimator from accessions.")

    p.add_argument("--sample-psms", default=None, type=int, help="Randomly sample this many PSMs from Sage results for analysis/plots.")

    # MS2Rescore options
    p.add_argument("--run-ms2rescore", action="store_true", help="Run MS2Rescore for rescoring")
    p.add_argument("--ms2rescore-bin", default="ms2rescore", help="MS2Rescore executable (default: ms2rescore).")
    p.add_argument("--ms2rescore-config", default=None, help="MS2Rescore config file (JSON/TOML). Optional.")
    p.add_argument("--ms2rescore-processes", type=int, default=0, help="Parallel processes for MS2Rescore. 0 = don't pass flag.")
    p.add_argument("--ms2rescore-spectrum-path", default=None, help="Spectrum file or directory for MS2Rescore. Default: common parent of --spectra.")
    p.add_argument("--dry-run", action="store_true", help="Print commands, but do not execute or analyze.")

    # Oktoberfest options
    p.add_argument("--run-oktoberfest", action="store_true", help="Run Oktoberfest for rescoring")

    # DIA-NN options
    p.add_argument("--diann-bin", default="diann", help="DIA-NN executable (default: diann).")
    p.add_argument("--diann-lib", default=None, help="Spectral library for DIA-NN (--lib). Optional.")
    p.add_argument("--diann-fasta-search", action="store_true", help="Enable DIA-NN library-free mode (--fasta-search).")
    p.add_argument("--diann-predictor", action="store_true", help="Enable DIA-NN deep-learning predictor (--predictor).")
    p.add_argument("--diann-gen-spec-lib", action="store_true", help="Generate spectral library (--gen-spec-lib).")
    p.add_argument("--diann-threads", type=int, default=None, help="Number of threads for DIA-NN.")
    p.add_argument("--diann-qvalue", type=float, default=None, help="Q-value filtering threshold for DIA-NN.")
    p.add_argument("--diann-extra-args", nargs="*", default=None, help="Additional raw CLI tokens for DIA-NN (e.g. --cut K*,R* --missed-cleavages 2).")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    spectra = [Path(x) for x in args.spectra]
    target_fasta = Path(args.target_fasta)
    out_dir = Path(args.output_dir)

    search_engine = args.search_engine
    fragmentation = args.fragmentation
    protease = args.protease
    protease_cfg = PROTEASE_CONFIGS[protease]

    # Apply HLA length overrides from CLI
    if protease == "hla":
        from dataclasses import replace as _dc_replace
        protease_cfg = _dc_replace(protease_cfg, min_len=args.hla_min_len, max_len=args.hla_max_len)

    entrap_cfg = EntrapmentConfig(
        mode=args.entrapment_mode,
        decoy_prefix=args.decoy_prefix,
        entrapment_fasta=Path(args.entrapment_fasta) if args.entrapment_fasta else None,
        entrapment_prefix=args.entrapment_prefix,
        entrapment_strategy=args.entrapment_strategy,
        pairing_regex=args.pairing_regex,
        paired_seed=args.paired_seed,
        paired_keep_c_terminal=bool(args.paired_keep_c_terminal),
        paired_min_len=int(args.paired_min_len) if protease != "hla" else protease_cfg.min_len,
        paired_max_len=int(args.paired_max_len) if protease != "hla" else protease_cfg.max_len,
        paired_missed_cleavages=int(args.paired_missed_cleavages),
        paired_digest_cleave_at=protease_cfg.cleave_at,
        paired_digest_restrict=protease_cfg.restrict,
        paired_digest_c_terminal=protease_cfg.c_terminal,
        collapse_il=bool(args.collapse_il),
    )

    # Validate: ms2rescore and oktoberfest are sage-only
    if search_engine == "diann":
        if args.run_ms2rescore:
            print("Error: --run-ms2rescore is only supported with --search-engine sage", file=sys.stderr)
            return 2
        if args.run_oktoberfest:
            print("Error: --run-oktoberfest is only supported with --search-engine sage", file=sys.stderr)
            return 2

    # Validate: ETD fragmentation compatibility
    if fragmentation == "etd":
        if search_engine == "diann":
            print("Error: --fragmentation etd is not supported with --search-engine diann", file=sys.stderr)
            return 2
        if args.run_ms2rescore:
            print("Error: --fragmentation etd is not supported with --run-ms2rescore", file=sys.stderr)
            return 2
        if args.run_oktoberfest:
            print("Error: --fragmentation etd is not supported with --run-oktoberfest", file=sys.stderr)
            return 2

    inputs = PipelineInputs(
        spectra=spectra,
        target_fasta=target_fasta,
        output_dir=out_dir,
        entrap_cfg=entrap_cfg,
        search_engine=search_engine,
        fragmentation=fragmentation,
        protease=protease,
        protease_cfg=protease_cfg,
        sage_bin=args.sage_bin,
        experiment_name=args.experiment_name,
        sage_config_in=Path(args.sage_config) if args.sage_config else None,
        ms2rescore_bin=args.ms2rescore_bin,
        ms2rescore_config=Path(args.ms2rescore_config) if args.ms2rescore_config else None,
        run_ms2rescore=bool(args.run_ms2rescore),
        ms2rescore_processes=int(args.ms2rescore_processes),
        spectrum_path_for_ms2rescore=Path(args.ms2rescore_spectrum_path) if args.ms2rescore_spectrum_path else None,
        run_oktoberfest=bool(args.run_oktoberfest),
        dry_run=bool(args.dry_run),
        sample_psms=int(args.sample_psms) if args.sample_psms else None,
        diann_bin=args.diann_bin,
        diann_lib=Path(args.diann_lib) if args.diann_lib else None,
        diann_fasta_search=bool(args.diann_fasta_search),
        diann_predictor=bool(args.diann_predictor),
        diann_gen_spec_lib=bool(args.diann_gen_spec_lib),
        diann_threads=args.diann_threads,
        diann_qvalue=args.diann_qvalue,
        diann_extra_args=args.diann_extra_args,
    )

    if entrap_cfg.mode == "foreign" and entrap_cfg.entrapment_fasta is None:
        print("Error: --entrapment-mode foreign requires --entrapment-fasta", file=sys.stderr)
        return 2

    run_pipeline(inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
