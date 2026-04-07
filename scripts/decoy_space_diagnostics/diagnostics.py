#!/usr/bin/env python3
"""
Decoy diagnostics for peptide databases.
Main features
-------------
- Digests a FASTA with targets and decoys (decoys identified by a header tag).
- If a peptide sequence appears in both target and decoy sets, it is treated as target-only
  for all competition analyses. The overlap is still reported explicitly.
- Computes non-stealability diagnostics from target queries:
    * best decoy shared theoretical b/y fragment count in a precursor window
    * best decoy spectral cosine in a precursor window
- Computes null diagnostics from target and decoy queries:
    * best false-target score vs best decoy score
    * expected target win rate from local target/decoy competitor counts
- Optional synthetic random-null queries.
- Writes both absolute-count plots (*.png) and normalized relative plots (*_relative.png).
- Supports process/thread workers for the heavy query-vs-library sweeps.
This script is designed to be self-contained and not require pyteomics/scipy.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import gzip
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _plot_ext() -> str:
    """Return '.pdf' if PLOT_PDFS=1, else '.png'."""
    return ".pdf" if os.environ.get("PLOT_PDFS") == "1" else ".png"

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _DummyTqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)
        def update(self, n: int = 1) -> None:
            return None
        def close(self) -> None:
            return None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb) -> None:
            return None
    def tqdm(iterable=None, *args, **kwargs):
        return _DummyTqdm(iterable=iterable, **kwargs)

# Monoisotopic residue masses for residues in a peptide chain (without H2O).
AA_MASS = {
    "A": 71.037113805,
    "R": 156.101111050,
    "N": 114.042927470,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.058577540,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "L": 113.084064015,
    "K": 128.094963050,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.079312980,
    "Y": 163.063328575,
    "V": 99.068413945,
}

PROTON = 1.007276466812
H2O = 18.01056468403

# Global worker state for fork-based process workers / threads.
_WORKER_STATE: dict = {}

@dataclass(frozen=True)
class Spectrum:
    mz: np.ndarray
    intensity: np.ndarray
    @property
    def n_peaks(self) -> int:
        return int(self.mz.size)

@dataclass(frozen=True)
class PeptideEntry:
    sequence: str
    label: str               # effective label: target / decoy
    source_class: str        # target_only / decoy_only / overlap_promoted_to_target
    mass: float
    length: int
    target_occurrences: int
    decoy_occurrences: int
    protein_roots: Tuple[str, ...]
    shannon_entropy: float
    fragment_mz: np.ndarray
    spectrum: Spectrum

@dataclass(frozen=True)
class ParallelPlan:
    backend: str
    workers: int
    batch_size: int
    mp_context: Optional[mp.context.BaseContext] = None

def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", file=sys.stderr, flush=True)

def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def open_textmaybe_gzip(path: str) -> Iterable[str]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                yield line
    else:
        with open(path, "rt") as handle:
            for line in handle:
                yield line

def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    header = None
    seq_parts: List[str] = []
    for raw in open_textmaybe_gzip(path):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_parts)
            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)

def choose_sep(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".csv"):
        return ","
    return "\t"

def canonical_protein_name(header: str, decoy_tag: str) -> str:
    token = str(header).strip().split()[0]
    tag = str(decoy_tag)
    if token.startswith(tag):
        return token[len(tag):]
    return token

def shannon_entropy(seq: str) -> float:
    if not seq:
        return float("nan")
    counts = {}
    for aa in seq:
        counts[aa] = counts.get(aa, 0) + 1
    n = float(len(seq))
    h = 0.0
    for cnt in counts.values():
        p = cnt / n
        h -= p * math.log2(p)
    return float(h)

def peptide_mass(seq: str) -> float:
    return H2O + sum(AA_MASS[a] for a in seq)

def digest_trypsin(sequence: str, missed_cleavages: int, min_length: int, max_length: int) -> List[str]:
    cut_sites = [0]
    n = len(sequence)
    for i in range(n - 1):
        aa = sequence[i]
        nxt = sequence[i + 1]
        if aa in {"K", "R"} and nxt != "P":
            cut_sites.append(i + 1)
    cut_sites.append(n)
    peptides: List[str] = []
    for start_idx in range(len(cut_sites) - 1):
        for mc in range(missed_cleavages + 1):
            stop_idx = start_idx + mc + 1
            if stop_idx >= len(cut_sites):
                continue
            pep = sequence[cut_sites[start_idx]:cut_sites[stop_idx]]
            if min_length <= len(pep) <= max_length:
                peptides.append(pep)
    return peptides

def theoretical_fragment_mz(seq: str, fragment_charges: Sequence[int]) -> np.ndarray:
    if len(seq) <= 1:
        return np.array([], dtype=np.float64)
    prefix_mass = 0.0
    prefix_neutral: List[float] = []
    for aa in seq[:-1]:
        prefix_mass += AA_MASS[aa]
        prefix_neutral.append(prefix_mass)
    total_residue_mass = sum(AA_MASS[a] for a in seq)
    out: List[float] = []
    for pm in prefix_neutral:
        suffix = total_residue_mass - pm
        for z in fragment_charges:
            if z <= 0:
                continue
            b_mz = (pm + z * PROTON) / z
            y_mz = (suffix + H2O + z * PROTON) / z
            out.append(b_mz)
            out.append(y_mz)
    arr = np.array(sorted(out), dtype=np.float64)
    return arr

def normalize_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    top_k: Optional[int] = None,
    min_intensity: float = 0.0,
) -> Spectrum:
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    mask = np.isfinite(mz) & np.isfinite(intensity) & (intensity > min_intensity)
    mz = mz[mask]
    intensity = intensity[mask]
    if mz.size == 0:
        return Spectrum(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    order = np.argsort(mz)
    mz = mz[order]
    intensity = intensity[order]
    if top_k is not None and mz.size > top_k:
        keep = np.argsort(intensity)[-top_k:]
        keep = np.sort(keep)
        mz = mz[keep]
        intensity = intensity[keep]
    norm = np.linalg.norm(intensity)
    if norm <= 0.0:
        return Spectrum(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    intensity = intensity / norm
    order = np.argsort(mz)
    return Spectrum(mz[order], intensity[order])

def theoretical_stick_spectrum(fragment_mz: np.ndarray, *, top_k: Optional[int] = None) -> Spectrum:
    if fragment_mz.size == 0:
        return Spectrum(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    return normalize_spectrum(fragment_mz, np.ones_like(fragment_mz), top_k=top_k, min_intensity=0.0)

def count_matches_with_tolerance(a: np.ndarray, b: np.ndarray, tol: float) -> int:
    i = 0
    j = 0
    n = 0
    while i < a.size and j < b.size:
        d = a[i] - b[j]
        if abs(d) <= tol:
            n += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return n

def sparse_cosine(spec_a: Spectrum, spec_b: Spectrum, tol: float) -> float:
    if spec_a.n_peaks == 0 or spec_b.n_peaks == 0:
        return float("nan")
    i = 0
    j = 0
    s = 0.0
    a_mz = spec_a.mz
    a_i = spec_a.intensity
    b_mz = spec_b.mz
    b_i = spec_b.intensity
    while i < a_mz.size and j < b_mz.size:
        d = a_mz[i] - b_mz[j]
        if abs(d) <= tol:
            s += float(a_i[i] * b_i[j])
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return max(0.0, min(1.0, s))

def _reshape_koina_output(obj: dict) -> np.ndarray:
    data = obj.get("data", [])
    shape = obj.get("shape", None)
    arr = np.asarray(data)
    if shape:
        try:
            arr = arr.reshape(shape)
        except Exception:
            pass
    return arr

def load_predicted_spectra(
    path: str,
    *,
    precursor_charge: Optional[int],
    collision_energy: Optional[float],
    top_k: Optional[int],
    min_intensity: float,
) -> Dict[str, Spectrum]:
    sep = choose_sep(path)
    df = pd.read_csv(path, sep=sep)
    lower = {c.lower(): c for c in df.columns}
    pep_col = lower.get("peptide", lower.get("sequence"))
    mz_col = lower.get("mz")
    intensity_col = lower.get("intensity")
    if pep_col is None or mz_col is None or intensity_col is None:
        raise ValueError("Predicted spectra file must contain peptide/sequence, mz, intensity columns.")
    if precursor_charge is not None and "precursor_charge" in lower:
        df = df[df[lower["precursor_charge"]] == precursor_charge]
    if collision_energy is not None and "collision_energy" in lower:
        ce = df[lower["collision_energy"]].astype(float).to_numpy()
        df = df[np.isclose(ce, float(collision_energy), atol=1e-6)]
    spectra: Dict[str, Spectrum] = {}
    groups = df.groupby(pep_col, sort=False)
    for peptide, grp in tqdm(groups, total=int(df[pep_col].nunique()), desc="Normalizing predicted spectra", unit="peptide"):
        spectra[str(peptide)] = normalize_spectrum(
            grp[mz_col].to_numpy(dtype=float),
            grp[intensity_col].to_numpy(dtype=float),
            top_k=top_k,
            min_intensity=min_intensity,
        )
    return spectra

def fetch_koina_predictions(
    peptides: Sequence[str],
    *,
    url: str,
    model_name: str,
    precursor_charge: int,
    collision_energy: float,
    batch_size: int,
    top_k: Optional[int],
    min_intensity: float,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
) -> Dict[str, Spectrum]:
    endpoint = url.rstrip("/") + f"/v2/models/{model_name}/infer"
    out: Dict[str, Spectrum] = {}
    total_batches = max(1, math.ceil(len(peptides) / batch_size))
    for bstart in tqdm(range(0, len(peptides), batch_size), total=total_batches, desc="Koina inference", unit="batch"):
        batch = list(peptides[bstart:bstart + batch_size])
        payload = {
            "id": f"batch_{bstart}",
            "inputs": [
                {"name": "peptide_sequences", "shape": [len(batch), 1], "datatype": "BYTES", "data": batch},
                {"name": "collision_energies", "shape": [len(batch), 1], "datatype": "FP32", "data": [float(collision_energy)] * len(batch)},
                {"name": "precursor_charges", "shape": [len(batch), 1], "datatype": "INT32", "data": [int(precursor_charge)] * len(batch)},
            ],
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response_json = None
        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    response_json = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if attempt == max_retries:
                    raise RuntimeError(f"Koina request failed ({exc.code}): {detail}") from exc
                time.sleep(sleep_seconds * attempt)
            except Exception as exc:
                if attempt == max_retries:
                    raise RuntimeError(f"Koina request failed: {exc}") from exc
                time.sleep(sleep_seconds * attempt)
        assert response_json is not None
        outputs = {obj["name"]: _reshape_koina_output(obj) for obj in response_json["outputs"]}
        if not {"mz", "intensities"}.issubset(outputs):
            raise RuntimeError("Koina response missing mz/intensities.")
        mz_mat = np.asarray(outputs["mz"], dtype=np.float64)
        int_mat = np.asarray(outputs["intensities"], dtype=np.float64)
        for peptide, mz_row, int_row in zip(batch, mz_mat, int_mat):
            out[peptide] = normalize_spectrum(mz_row, int_row, top_k=top_k, min_intensity=min_intensity)
    return out

def write_spectra_cache(path: pathlib.Path, spectra: Dict[str, Spectrum]) -> None:
    rows: List[dict] = []
    for peptide, spec in spectra.items():
        for mz, inten in zip(spec.mz, spec.intensity):
            rows.append({"peptide": peptide, "mz": float(mz), "intensity": float(inten)})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)

def load_spectra_cache(path: pathlib.Path) -> Dict[str, Spectrum]:
    df = pd.read_csv(path, sep="\t")
    spectra: Dict[str, Spectrum] = {}
    for peptide, grp in df.groupby("peptide", sort=False):
        mz = grp["mz"].to_numpy(dtype=float)
        intensity = grp["intensity"].to_numpy(dtype=float)
        spectra[str(peptide)] = Spectrum(mz=mz, intensity=intensity, n_peaks=len(mz))
    return spectra

def build_peptide_table(
    fasta_path: str,
    *,
    decoy_tag: str,
    missed_cleavages: int,
    min_length: int,
    max_length: int,
) -> Tuple[pd.DataFrame, dict]:
    target_counts: Dict[str, int] = {}
    decoy_counts: Dict[str, int] = {}
    target_proteins: Dict[str, set[str]] = {}
    decoy_proteins: Dict[str, set[str]] = {}
    protein_counts = {"target": 0, "decoy": 0}
    skipped_unknown = 0
    for header, sequence in tqdm(read_fasta(fasta_path), desc="Digesting FASTA", unit="protein"):
        label = "decoy" if decoy_tag.lower() in header.lower() else "target"
        protein_counts[label] += 1
        root = canonical_protein_name(header, decoy_tag)
        seq = sequence.strip().upper()
        peptides = digest_trypsin(seq, missed_cleavages=missed_cleavages, min_length=min_length, max_length=max_length)
        dest_counts = decoy_counts if label == "decoy" else target_counts
        dest_proteins = decoy_proteins if label == "decoy" else target_proteins
        for pep in peptides:
            if any(a not in AA_MASS for a in pep):
                skipped_unknown += 1
                continue
            dest_counts[pep] = dest_counts.get(pep, 0) + 1
            roots = dest_proteins.get(pep)
            if roots is None:
                roots = set()
                dest_proteins[pep] = roots
            roots.add(root)
    all_peptides = sorted(set(target_counts) | set(decoy_counts))
    rows: List[dict] = []
    for pep in tqdm(all_peptides, desc="Assembling unique peptide table", unit="peptide"):
        t_occ = int(target_counts.get(pep, 0))
        d_occ = int(decoy_counts.get(pep, 0))
        t_roots = tuple(sorted(target_proteins.get(pep, set())))
        d_roots = tuple(sorted(decoy_proteins.get(pep, set())))
        overlap = bool(t_occ > 0 and d_occ > 0)
        if t_occ > 0:
            eff_label = "target"
            source_class = "overlap_promoted_to_target" if overlap else "target_only"
            eff_roots = t_roots
        else:
            eff_label = "decoy"
            source_class = "decoy_only"
            eff_roots = d_roots
        rows.append(
            {
                "sequence": pep,
                "length": len(pep),
                "mass": peptide_mass(pep),
                "shannon_entropy": shannon_entropy(pep),
                "target_occurrences": t_occ,
                "decoy_occurrences": d_occ,
                "total_occurrences": t_occ + d_occ,
                "target_protein_roots": ";".join(t_roots),
                "decoy_protein_roots": ";".join(d_roots),
                "n_target_protein_roots": len(t_roots),
                "n_decoy_protein_roots": len(d_roots),
                "effective_protein_roots": ";".join(eff_roots),
                "n_effective_protein_roots": len(eff_roots),
                "is_target_decoy_overlap": overlap,
                "effective_label": eff_label,
                "source_class": source_class,
            }
        )
    peptide_df = pd.DataFrame(rows)
    meta = {
        "n_target_proteins": protein_counts["target"],
        "n_decoy_proteins": protein_counts["decoy"],
        "n_unique_sequences": int(len(peptide_df)),
        "n_effective_targets": int((peptide_df["effective_label"] == "target").sum()),
        "n_effective_decoys": int((peptide_df["effective_label"] == "decoy").sum()),
        "n_target_decoy_overlaps": int(peptide_df["is_target_decoy_overlap"].sum()),
        "skipped_unknown_peptides": int(skipped_unknown),
    }
    return peptide_df, meta

def write_overlap_reports(peptide_df: pd.DataFrame, outdir: pathlib.Path) -> dict:
    overlap_df = peptide_df[peptide_df["is_target_decoy_overlap"]].copy()
    overlap_df = overlap_df.sort_values(["length", "mass", "sequence"]).reset_index(drop=True)
    overlap_path = outdir / "target_decoy_sequence_overlaps.tsv"
    overlap_df.to_csv(overlap_path, sep="\t", index=False)
    with open(outdir / "exact_target_decoy_sequence_overlaps.txt", "wt") as handle:
        if overlap_df.empty:
            handle.write("No exact target/decoy sequence overlaps were detected.\n")
        else:
            handle.write("# Overlap peptides are promoted to target and removed from the effective decoy pool.\n")
            for seq in overlap_df["sequence"]:
                handle.write(f"{seq}\n")
    return {"n_target_decoy_overlaps": int(len(overlap_df))}

def build_entries(
    peptide_df: pd.DataFrame,
    *,
    fragment_charges: Sequence[int],
    spectra: Dict[str, Spectrum],
    pred_top_k: Optional[int],
) -> Tuple[List[PeptideEntry], List[PeptideEntry], dict]:
    seqs = peptide_df["sequence"].tolist()
    frag_cache: Dict[str, np.ndarray] = {}
    for seq in tqdm(seqs, desc="Computing theoretical fragments", unit="peptide"):
        frag_cache[seq] = theoretical_fragment_mz(seq, fragment_charges=fragment_charges)
    target_entries: List[PeptideEntry] = []
    decoy_entries: List[PeptideEntry] = []
    n_theoretical = 0
    for rec in tqdm(peptide_df.itertuples(index=False), total=len(peptide_df), desc="Building peptide entries", unit="peptide"):
        frag = frag_cache[rec.sequence]
        spec = spectra.get(rec.sequence)
        if spec is None:
            spec = theoretical_stick_spectrum(frag, top_k=pred_top_k)
            n_theoretical += 1
        if rec.effective_label == "target":
            roots = tuple(x for x in str(rec.target_protein_roots).split(";") if x)
        else:
            roots = tuple(x for x in str(rec.decoy_protein_roots).split(";") if x)
        entry = PeptideEntry(
            sequence=rec.sequence,
            label=rec.effective_label,
            source_class=rec.source_class,
            mass=float(rec.mass),
            length=int(rec.length),
            target_occurrences=int(rec.target_occurrences),
            decoy_occurrences=int(rec.decoy_occurrences),
            protein_roots=roots,
            shannon_entropy=float(rec.shannon_entropy),
            fragment_mz=frag,
            spectrum=spec,
        )
        if entry.label == "target":
            target_entries.append(entry)
        else:
            decoy_entries.append(entry)
    meta = {"n_theoretical_spectrum_fallbacks": int(n_theoretical)}
    return target_entries, decoy_entries, meta

class MassIndex:
    def __init__(self, entries: Sequence[PeptideEntry]):
        self.entries = list(entries)
        self.masses = np.asarray([x.mass for x in self.entries], dtype=np.float64)
        self.order = np.argsort(self.masses)
        self.sorted_masses = self.masses[self.order]
    def window(self, mass: float, width: float) -> np.ndarray:
        left = np.searchsorted(self.sorted_masses, mass - width, side="left")
        right = np.searchsorted(self.sorted_masses, mass + width, side="right")
        return self.order[left:right]

def maybe_sample_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=sample_size, replace=False))

def resolve_parallel_plan(workers: int, backend: str, batch_size: int) -> ParallelPlan:
    workers = max(1, int(workers))
    batch_size = max(1, int(batch_size))
    if workers <= 1 or backend == "none":
        return ParallelPlan(backend="none", workers=1, batch_size=batch_size, mp_context=None)
    chosen = backend
    if chosen == "auto":
        chosen = "process" if os.name == "posix" else "thread"
    if chosen == "process":
        if os.name == "posix":
            try:
                ctx = mp.get_context("fork")
                return ParallelPlan(backend="process", workers=workers, batch_size=batch_size, mp_context=ctx)
            except Exception:
                log("fork context unavailable, falling back to threads.")
        else:
            log("process backend not available here, falling back to threads.")
        chosen = "thread"
    if chosen == "thread":
        return ParallelPlan(backend="thread", workers=workers, batch_size=batch_size, mp_context=None)
    return ParallelPlan(backend="none", workers=1, batch_size=batch_size, mp_context=None)

def set_worker_state(
    target_entries: Sequence[PeptideEntry],
    decoy_entries: Sequence[PeptideEntry],
    *,
    mass_window: float,
    fragment_tol: float,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "targets": list(target_entries),
        "decoys": list(decoy_entries),
        "target_index": MassIndex(target_entries),
        "decoy_index": MassIndex(decoy_entries),
        "mass_window": float(mass_window),
        "fragment_tol": float(fragment_tol),
    }

def chunk_indices(indices: np.ndarray, batch_size: int) -> List[np.ndarray]:
    if indices.size == 0:
        return []
    return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

def run_parallel_batches(batches: Sequence, worker_fn, *, plan: ParallelPlan, desc: str) -> List[dict]:
    if not batches:
        return []
    if plan.backend == "none" or plan.workers <= 1 or len(batches) == 1:
        rows: List[dict] = []
        for batch in tqdm(batches, desc=desc, total=len(batches), unit="batch"):
            rows.extend(worker_fn(batch))
        return rows
    executor_cls = cf.ThreadPoolExecutor if plan.backend == "thread" else cf.ProcessPoolExecutor
    kwargs = {}
    if plan.backend == "process" and plan.mp_context is not None:
        kwargs["mp_context"] = plan.mp_context
    rows: List[dict] = []
    with executor_cls(max_workers=plan.workers, **kwargs) as ex:
        for part in tqdm(ex.map(worker_fn, batches), total=len(batches), desc=desc, unit="batch"):
            rows.extend(part)
    return rows

def classify_winner(target_score: float, decoy_score: float) -> str:
    if not np.isfinite(target_score) and not np.isfinite(decoy_score):
        return "missing"
    if np.isfinite(target_score) and not np.isfinite(decoy_score):
        return "target"
    if np.isfinite(decoy_score) and not np.isfinite(target_score):
        return "decoy"
    if target_score > decoy_score:
        return "target"
    if decoy_score > target_score:
        return "decoy"
    return "tie"

def _score_better(score: float, best_score: float, mass_delta: float, best_mass_delta: float, seq: str, best_seq: Optional[str]) -> bool:
    if not np.isfinite(score):
        return False
    if not np.isfinite(best_score):
        return True
    if score > best_score:
        return True
    if score < best_score:
        return False
    if mass_delta < best_mass_delta:
        return True
    if mass_delta > best_mass_delta:
        return False
    if best_seq is None:
        return True
    return seq < best_seq

def score_against_candidates(
    query: PeptideEntry,
    candidate_entries: Sequence[PeptideEntry],
    candidate_ids: np.ndarray,
    *,
    fragment_tol: float,
    skip_index: Optional[int] = None,
) -> dict:
    best_shared = float("nan")
    best_shared_seq: Optional[str] = None
    best_shared_mass_delta = float("inf")
    best_shared_same_protein = None
    best_shared_entropy = float("nan")
    best_cos = float("nan")
    best_cos_seq: Optional[str] = None
    best_cos_mass_delta = float("inf")
    best_cos_same_protein = None
    best_cos_entropy = float("nan")
    query_roots = set(query.protein_roots)
    for idx in candidate_ids:
        j = int(idx)
        if skip_index is not None and j == skip_index:
            continue
        cand = candidate_entries[j]
        mass_delta = abs(query.mass - cand.mass)
        same_protein = bool(query_roots.intersection(cand.protein_roots))
        shared = float(count_matches_with_tolerance(query.fragment_mz, cand.fragment_mz, fragment_tol))
        cos = sparse_cosine(query.spectrum, cand.spectrum, fragment_tol)
        if _score_better(shared, best_shared, mass_delta, best_shared_mass_delta, cand.sequence, best_shared_seq):
            best_shared = shared
            best_shared_seq = cand.sequence
            best_shared_mass_delta = mass_delta
            best_shared_same_protein = same_protein
            best_shared_entropy = cand.shannon_entropy
        if _score_better(cos, best_cos, mass_delta, best_cos_mass_delta, cand.sequence, best_cos_seq):
            best_cos = cos
            best_cos_seq = cand.sequence
            best_cos_mass_delta = mass_delta
            best_cos_same_protein = same_protein
            best_cos_entropy = cand.shannon_entropy
    return {
        "best_shared": best_shared,
        "best_shared_seq": best_shared_seq,
        "best_shared_mass_delta": (best_shared_mass_delta if np.isfinite(best_shared) else float("nan")),
        "best_shared_same_protein": best_shared_same_protein,
        "best_shared_entropy": best_shared_entropy,
        "best_cos": best_cos,
        "best_cos_seq": best_cos_seq,
        "best_cos_mass_delta": (best_cos_mass_delta if np.isfinite(best_cos) else float("nan")),
        "best_cos_same_protein": best_cos_same_protein,
        "best_cos_entropy": best_cos_entropy,
    }

def _same_protein_category(value: Optional[bool], has_score: bool) -> str:
    if not has_score:
        return "missing"
    return "same_protein_cross_version" if bool(value) else "other_protein"

def _null_worker(task: Tuple[str, np.ndarray]) -> List[dict]:
    query_label, batch = task
    state = _WORKER_STATE
    targets: Sequence[PeptideEntry] = state["targets"]
    decoys: Sequence[PeptideEntry] = state["decoys"]
    target_index: MassIndex = state["target_index"]
    decoy_index: MassIndex = state["decoy_index"]
    mass_window = float(state["mass_window"])
    fragment_tol = float(state["fragment_tol"])
    rows: List[dict] = []
    source_entries = targets if query_label == "target" else decoys
    for i in batch:
        q_idx = int(i)
        q = source_entries[q_idx]
        target_ids = target_index.window(q.mass, mass_window)
        decoy_ids = decoy_index.window(q.mass, mass_window)
        skip_t = q_idx if query_label == "target" else None
        skip_d = q_idx if query_label == "decoy" else None
        target_res = score_against_candidates(q, targets, target_ids, fragment_tol=fragment_tol, skip_index=skip_t)
        decoy_res = score_against_candidates(q, decoys, decoy_ids, fragment_tol=fragment_tol, skip_index=skip_d)
        best_t_shared = target_res["best_shared"]
        best_d_shared = decoy_res["best_shared"]
        best_t_cos = target_res["best_cos"]
        best_d_cos = decoy_res["best_cos"]
        n_false_targets = int(len(target_ids) - (1 if query_label == "target" else 0))
        n_decoys = int(len(decoy_ids) - (1 if query_label == "decoy" else 0))
        n_false_targets = max(0, n_false_targets)
        n_decoys = max(0, n_decoys)
        total = n_false_targets + n_decoys
        expected_target_win_rate = float(n_false_targets / total) if total > 0 else float("nan")
        local_decoy_fraction = float(n_decoys / total) if total > 0 else float("nan")
        has_false_target_shared = np.isfinite(best_t_shared)
        has_decoy_shared = np.isfinite(best_d_shared)
        has_false_target_cos = np.isfinite(best_t_cos)
        has_decoy_cos = np.isfinite(best_d_cos)
        rows.append(
            {
                "query_label": query_label,
                "query_sequence": q.sequence,
                "query_length": q.length,
                "query_mass": q.mass,
                "query_source_class": q.source_class,
                "query_entropy": q.shannon_entropy,
                "query_protein_roots": ";".join(q.protein_roots),
                "query_fragment_count": int(q.fragment_mz.size),
                "query_spectrum_peak_count": int(q.spectrum.n_peaks),
                "n_false_target_competitors": n_false_targets,
                "n_decoy_competitors": n_decoys,
                "local_decoy_fraction": local_decoy_fraction,
                "expected_target_win_rate": expected_target_win_rate,
                "best_false_target_shared": best_t_shared,
                "best_decoy_shared": best_d_shared,
                "delta_shared_target_minus_decoy": (
                    best_t_shared - best_d_shared if has_false_target_shared and has_decoy_shared else float("nan")
                ),
                "winner_shared": classify_winner(best_t_shared, best_d_shared),
                "best_false_target_sequence_shared": target_res["best_shared_seq"],
                "best_decoy_sequence_shared": decoy_res["best_shared_seq"],
                "best_false_target_mass_delta_shared": target_res["best_shared_mass_delta"],
                "best_decoy_mass_delta_shared": decoy_res["best_shared_mass_delta"],
                "best_false_target_same_protein_cross_version_shared": target_res["best_shared_same_protein"],
                "best_decoy_same_protein_cross_version_shared": decoy_res["best_shared_same_protein"],
                "best_false_target_same_protein_cross_version_shared_category": _same_protein_category(target_res["best_shared_same_protein"], has_false_target_shared),
                "best_decoy_same_protein_cross_version_shared_category": _same_protein_category(decoy_res["best_shared_same_protein"], has_decoy_shared),
                "best_false_target_entropy_shared": target_res["best_shared_entropy"],
                "best_decoy_entropy_shared": decoy_res["best_shared_entropy"],
                "best_false_target_cosine": best_t_cos,
                "best_decoy_cosine": best_d_cos,
                "delta_cosine_target_minus_decoy": (
                    best_t_cos - best_d_cos if has_false_target_cos and has_decoy_cos else float("nan")
                ),
                "winner_cosine": classify_winner(best_t_cos, best_d_cos),
                "best_false_target_sequence_cosine": target_res["best_cos_seq"],
                "best_decoy_sequence_cosine": decoy_res["best_cos_seq"],
                "best_false_target_mass_delta_cosine": target_res["best_cos_mass_delta"],
                "best_decoy_mass_delta_cosine": decoy_res["best_cos_mass_delta"],
                "best_false_target_same_protein_cross_version_cosine": target_res["best_cos_same_protein"],
                "best_decoy_same_protein_cross_version_cosine": decoy_res["best_cos_same_protein"],
                "best_false_target_same_protein_cross_version_cosine_category": _same_protein_category(target_res["best_cos_same_protein"], has_false_target_cos),
                "best_decoy_same_protein_cross_version_cosine_category": _same_protein_category(decoy_res["best_cos_same_protein"], has_decoy_cos),
                "best_false_target_entropy_cosine": target_res["best_cos_entropy"],
                "best_decoy_entropy_cosine": decoy_res["best_cos_entropy"],
            }
        )
    return rows

def compute_null_queries(
    target_entries: Sequence[PeptideEntry],
    decoy_entries: Sequence[PeptideEntry],
    *,
    mass_window: float,
    fragment_tol: float,
    sample_targets: int,
    sample_decoys: int,
    seed: int,
    plan: ParallelPlan,
) -> pd.DataFrame:
    set_worker_state(target_entries, decoy_entries, mass_window=mass_window, fragment_tol=fragment_tol)
    tasks: List[Tuple[str, np.ndarray]] = []
    tgt_idx = maybe_sample_indices(len(target_entries), sample_targets, seed)
    dcy_idx = maybe_sample_indices(len(decoy_entries), sample_decoys, seed + 1)
    tasks.extend([("target", x) for x in chunk_indices(tgt_idx, plan.batch_size)])
    tasks.extend([("decoy", x) for x in chunk_indices(dcy_idx, plan.batch_size)])
    rows = run_parallel_batches(tasks, _null_worker, plan=plan, desc="Null/non-stealability sweeps")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["query_label", "query_mass", "query_sequence"]).reset_index(drop=True)
    return df

def derive_nonstealability(null_df: pd.DataFrame) -> pd.DataFrame:
    df = null_df[null_df["query_label"] == "target"].copy()
    if df.empty:
        return df
    df["best_decoy_shared_fraction"] = df["best_decoy_shared"] / df["query_fragment_count"].replace(0, np.nan)
    df["best_decoy_cosine_distance"] = 1.0 - df["best_decoy_cosine"]
    df["n_target_competitors"] = df["n_false_target_competitors"]
    return df[
        [
            "query_sequence",
            "query_length",
            "query_mass",
            "query_source_class",
            "query_entropy",
            "query_protein_roots",
            "query_fragment_count",
            "query_spectrum_peak_count",
            "n_target_competitors",
            "n_decoy_competitors",
            "local_decoy_fraction",
            "best_decoy_shared",
            "best_decoy_shared_fraction",
            "best_decoy_sequence_shared",
            "best_decoy_mass_delta_shared",
            "best_decoy_same_protein_cross_version_shared",
            "best_decoy_same_protein_cross_version_shared_category",
            "best_decoy_entropy_shared",
            "best_decoy_cosine",
            "best_decoy_cosine_distance",
            "best_decoy_sequence_cosine",
            "best_decoy_mass_delta_cosine",
            "best_decoy_same_protein_cross_version_cosine",
            "best_decoy_same_protein_cross_version_cosine_category",
            "best_decoy_entropy_cosine",
        ]
    ].reset_index(drop=True)

def _random_null_worker(batch: List[Tuple[float, int, int]]) -> List[dict]:
    state = _WORKER_STATE
    targets: Sequence[PeptideEntry] = state["targets"]
    decoys: Sequence[PeptideEntry] = state["decoys"]
    target_index: MassIndex = state["target_index"]
    decoy_index: MassIndex = state["decoy_index"]
    mass_window = float(state["mass_window"])
    fragment_tol = float(state["fragment_tol"])
    rows: List[dict] = []
    for mass, n_peaks, seed_val in batch:
        rng = np.random.default_rng(int(seed_val))
        mz = np.sort(rng.uniform(100.0, min(max(300.0, mass), 2200.0), size=int(n_peaks)))
        intensity = rng.exponential(scale=1.0, size=int(n_peaks))
        spec = normalize_spectrum(mz, intensity)
        frag = spec.mz
        target_ids = target_index.window(mass, mass_window)
        decoy_ids = decoy_index.window(mass, mass_window)
        best_t_shared = float("nan")
        best_t_cos = float("nan")
        best_d_shared = float("nan")
        best_d_cos = float("nan")
        best_t_s = -1.0
        best_d_s = -1.0
        best_t_c = -1.0
        best_d_c = -1.0
        for idx in target_ids:
            cand = targets[int(idx)]
            shared = float(count_matches_with_tolerance(frag, cand.fragment_mz, fragment_tol))
            cos = sparse_cosine(spec, cand.spectrum, fragment_tol)
            if shared > best_t_s:
                best_t_s = shared
                best_t_shared = shared
            if np.isfinite(cos) and cos > best_t_c:
                best_t_c = cos
                best_t_cos = cos
        for idx in decoy_ids:
            cand = decoys[int(idx)]
            shared = float(count_matches_with_tolerance(frag, cand.fragment_mz, fragment_tol))
            cos = sparse_cosine(spec, cand.spectrum, fragment_tol)
            if shared > best_d_s:
                best_d_s = shared
                best_d_shared = shared
            if np.isfinite(cos) and cos > best_d_c:
                best_d_c = cos
                best_d_cos = cos
        n_t = int(len(target_ids))
        n_d = int(len(decoy_ids))
        total = n_t + n_d
        rows.append(
            {
                "query_mass": float(mass),
                "query_peak_count": int(spec.n_peaks),
                "n_target_competitors": n_t,
                "n_decoy_competitors": n_d,
                "local_decoy_fraction": float(n_d / total) if total > 0 else float("nan"),
                "expected_target_win_rate": float(n_t / total) if total > 0 else float("nan"),
                "best_target_shared": best_t_shared,
                "best_decoy_shared": best_d_shared,
                "winner_shared": classify_winner(best_t_shared, best_d_shared),
                "best_target_cosine": best_t_cos,
                "best_decoy_cosine": best_d_cos,
                "winner_cosine": classify_winner(best_t_cos, best_d_cos),
            }
        )
    return rows

def simulate_random_null_queries(
    target_entries: Sequence[PeptideEntry],
    decoy_entries: Sequence[PeptideEntry],
    *,
    mass_window: float,
    fragment_tol: float,
    n_queries: int,
    seed: int,
    plan: ParallelPlan,
) -> pd.DataFrame:
    if n_queries <= 0:
        return pd.DataFrame()
    set_worker_state(target_entries, decoy_entries, mass_window=mass_window, fragment_tol=fragment_tol)
    rng = np.random.default_rng(seed)
    all_masses = np.asarray([x.mass for x in target_entries + decoy_entries], dtype=np.float64)
    all_peaks = np.asarray([max(1, x.spectrum.n_peaks) for x in target_entries + decoy_entries], dtype=int)
    masses = rng.choice(all_masses, size=n_queries, replace=True)
    peaks = rng.choice(all_peaks, size=n_queries, replace=True)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_queries)
    specs = [(float(m), int(k), int(s)) for m, k, s in zip(masses, peaks, seeds)]
    batches = [specs[i:i + plan.batch_size] for i in range(0, len(specs), plan.batch_size)]
    rows = run_parallel_batches(batches, _random_null_worker, plan=plan, desc="Random null sweeps")
    return pd.DataFrame(rows)

def wilson_interval(successes: float, total: float, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))

    return max(0.0, center - half), min(1.0, center + half)

def ks_2samp_basic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.sort(x)
    y = np.sort(y)
    values = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, values, side="right") / x.size
    cdf_y = np.searchsorted(y, values, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))

def safe_savefig(path: pathlib.Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def empty_plot(path: pathlib.Path, title: str, message: str = "No data") -> None:
    plt.figure(figsize=(8, 5))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.title(title)
    plt.axis("off")
    safe_savefig(path)

def finite_values(series: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    return arr[np.isfinite(arr)]

def plot_two_label_histograms(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    outbase: pathlib.Path,
    title: str,
    xlabel: str,
    bins: int | Sequence[float] = 50,
    labels: Tuple[str, str] = ("target", "decoy"),
) -> None:
    sub = df[[value_col, label_col]].copy()
    a = finite_values(sub.loc[sub[label_col] == labels[0], value_col])
    b = finite_values(sub.loc[sub[label_col] == labels[1], value_col])
    if a.size == 0 and b.size == 0:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    plt.figure(figsize=(8, 5))
    if a.size:
        plt.hist(a, bins=bins, alpha=0.6, label=labels[0])
    if b.size:
        plt.hist(b, bins=bins, alpha=0.6, label=labels[1])
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    safe_savefig(outbase.with_suffix(_plot_ext()))
    plt.figure(figsize=(8, 5))
    if a.size:
        plt.hist(a, bins=bins, alpha=0.6, label=labels[0], weights=np.ones_like(a) / max(1, a.size))
    if b.size:
        plt.hist(b, bins=bins, alpha=0.6, label=labels[1], weights=np.ones_like(b) / max(1, b.size))
    plt.xlabel(xlabel)
    plt.ylabel("Proportion")
    plt.title(title + " (relative)")
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    safe_savefig(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}")

def plot_single_histograms(
    values: np.ndarray,
    outbase: pathlib.Path,
    title: str,
    xlabel: str,
    bins: int | Sequence[float] = 50,
) -> None:
    vals = finite_values(values)
    if vals.size == 0:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    safe_savefig(outbase.with_suffix(_plot_ext()))
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=bins, weights=np.ones_like(vals) / max(1, vals.size))
    plt.xlabel(xlabel)
    plt.ylabel("Proportion")
    plt.title(title + " (relative)")
    safe_savefig(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}")

def plot_mass_bin_counts(peptide_df: pd.DataFrame, outdir: pathlib.Path) -> None:
    if peptide_df.empty:
        empty_plot(outdir / f"mass_bin_counts{_plot_ext()}", "Target/decoy mass-bin counts")
        empty_plot(outdir / f"mass_bin_counts_relative{_plot_ext()}", "Target/decoy mass-bin proportions")
        return
    bins = np.linspace(peptide_df["mass"].min(), peptide_df["mass"].max(), 80)
    t = peptide_df.loc[peptide_df["effective_label"] == "target", "mass"].to_numpy(dtype=float)
    d = peptide_df.loc[peptide_df["effective_label"] == "decoy", "mass"].to_numpy(dtype=float)
    t_counts, edges = np.histogram(t, bins=bins)
    d_counts, _ = np.histogram(d, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(9, 5))
    plt.plot(centers, t_counts, label="target")
    plt.plot(centers, d_counts, label="decoy")
    plt.xlabel("Neutral monoisotopic peptide mass (Da)")
    plt.ylabel("Count")
    plt.title("Target/decoy counts by precursor-mass bin")
    plt.legend()
    safe_savefig(outdir / f"mass_bin_counts{_plot_ext()}")
    plt.figure(figsize=(9, 5))
    if t_counts.sum() > 0:
        plt.plot(centers, t_counts / t_counts.sum(), label="target")
    if d_counts.sum() > 0:
        plt.plot(centers, d_counts / d_counts.sum(), label="decoy")
    plt.xlabel("Neutral monoisotopic peptide mass (Da)")
    plt.ylabel("Proportion")
    plt.title("Target/decoy proportions by precursor-mass bin")
    plt.legend()
    safe_savefig(outdir / f"mass_bin_counts_relative{_plot_ext()}")
    plt.figure(figsize=(9, 5))
    ratio = (d_counts + 1.0) / (t_counts + 1.0)
    plt.plot(centers, ratio)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Neutral monoisotopic peptide mass (Da)")
    plt.ylabel("(decoy + 1)/(target + 1)")
    plt.title("Decoy/target count ratio by mass bin")
    safe_savefig(outdir / f"mass_bin_decoy_target_ratio{_plot_ext()}")

def plot_mass_length_balance_heatmap(peptide_df: pd.DataFrame, outdir: pathlib.Path) -> None:
    if peptide_df.empty:
        empty_plot(outdir / f"mass_length_balance_heatmap{_plot_ext()}", "Peptide count by mass and length")
        empty_plot(outdir / f"mass_length_balance_heatmap_relative{_plot_ext()}", "Decoy/target ratio by mass and length")
        return
    length_edges = np.arange(peptide_df["length"].min(), peptide_df["length"].max() + 2) - 0.5
    mass_edges = np.linspace(peptide_df["mass"].min(), peptide_df["mass"].max(), 50)
    counts, _, _ = np.histogram2d(
        peptide_df["length"].to_numpy(dtype=float),
        peptide_df["mass"].to_numpy(dtype=float),
        bins=[length_edges, mass_edges],
    )
    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(mass_edges, length_edges, counts, shading="auto")
    cb = plt.colorbar(mesh)
    cb.set_label("Count")
    plt.xlabel("Neutral monoisotopic peptide mass (Da)")
    plt.ylabel("Peptide length")
    plt.title("All effective peptides by mass and length")
    safe_savefig(outdir / f"mass_length_balance_heatmap{_plot_ext()}")
    tdf = peptide_df[peptide_df["effective_label"] == "target"]
    ddf = peptide_df[peptide_df["effective_label"] == "decoy"]
    th, _, _ = np.histogram2d(tdf["length"].to_numpy(dtype=float), tdf["mass"].to_numpy(dtype=float), bins=[length_edges, mass_edges])
    dh, _, _ = np.histogram2d(ddf["length"].to_numpy(dtype=float), ddf["mass"].to_numpy(dtype=float), bins=[length_edges, mass_edges])
    ratio = np.log2((dh + 1.0) / (th + 1.0))
    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(mass_edges, length_edges, ratio, shading="auto")
    cb = plt.colorbar(mesh)
    cb.set_label("log2((decoy + 1)/(target + 1))")
    plt.xlabel("Neutral monoisotopic peptide mass (Da)")
    plt.ylabel("Peptide length")
    plt.title("Target/decoy balance by mass and length")
    safe_savefig(outdir / f"mass_length_balance_heatmap_relative{_plot_ext()}")

def plot_1d_heatmap_by_length(
    df: pd.DataFrame,
    *,
    metric_col: str,
    outbase: pathlib.Path,
    title: str,
    xlabel: str,
    metric_edges: np.ndarray,
) -> None:
    if df.empty or metric_col not in df:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    sub = df[["query_length", metric_col]].copy()
    sub = sub[np.isfinite(sub[metric_col])]
    if sub.empty:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    length_edges = np.arange(sub["query_length"].min(), sub["query_length"].max() + 2) - 0.5
    hist, _, _ = np.histogram2d(
        sub["query_length"].to_numpy(dtype=float),
        sub[metric_col].to_numpy(dtype=float),
        bins=[length_edges, metric_edges],
    )
    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(metric_edges, length_edges, hist, shading="auto")
    cb = plt.colorbar(mesh)
    cb.set_label("Count")
    plt.xlabel(xlabel)
    plt.ylabel("Query peptide length")
    plt.title(title)
    safe_savefig(outbase.with_suffix(_plot_ext()))
    row_sums = hist.sum(axis=1, keepdims=True)
    rel = np.divide(hist, row_sums, out=np.zeros_like(hist), where=row_sums > 0)
    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(metric_edges, length_edges, rel, shading="auto")
    cb = plt.colorbar(mesh)
    cb.set_label("Proportion within length")
    plt.xlabel(xlabel)
    plt.ylabel("Query peptide length")
    plt.title(title + " (relative)")
    safe_savefig(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}")

def plot_category_bars(
    df: pd.DataFrame,
    category_col: str,
    outbase: pathlib.Path,
    title: str,
    order: Optional[Sequence[str]] = None,
) -> None:
    if df.empty or category_col not in df:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    cats = df[category_col].fillna("missing").astype(str)
    if order is None:
        order = sorted(cats.unique().tolist())
    counts = np.asarray([(cats == name).sum() for name in order], dtype=float)
    x = np.arange(len(order))
    plt.figure(figsize=(8, 5))
    plt.bar(x, counts)
    plt.xticks(x, order, rotation=15, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    safe_savefig(outbase.with_suffix(_plot_ext()))
    total = max(1.0, float(counts.sum()))
    plt.figure(figsize=(8, 5))
    plt.bar(x, counts / total)
    plt.xticks(x, order, rotation=15, ha="right")
    plt.ylabel("Proportion")
    plt.title(title + " (relative)")
    safe_savefig(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}")

def plot_category_by_length(
    df: pd.DataFrame,
    category_col: str,
    outbase: pathlib.Path,
    title: str,
    order: Optional[Sequence[str]] = None,
) -> None:
    if df.empty or category_col not in df or "query_length" not in df:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    cats = df[category_col].fillna("missing").astype(str)
    if order is None:
        order = sorted(cats.unique().tolist())
    lengths = sorted(df["query_length"].dropna().astype(int).unique().tolist())
    if not lengths:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    table = np.zeros((len(order), len(lengths)), dtype=float)
    for li, length in enumerate(lengths):
        sub = df.loc[df["query_length"] == length, category_col].fillna("missing").astype(str)
        for oi, name in enumerate(order):
            table[oi, li] = float((sub == name).sum())
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(lengths), dtype=float)
    for oi, name in enumerate(order):
        plt.bar(lengths, table[oi], bottom=bottom, label=name)
        bottom += table[oi]
    plt.xlabel("Query peptide length")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    safe_savefig(outbase.with_suffix(_plot_ext()))
    sums = table.sum(axis=0, keepdims=True)
    rel = np.divide(table, sums, out=np.zeros_like(table), where=sums > 0)
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(lengths), dtype=float)
    for oi, name in enumerate(order):
        plt.bar(lengths, rel[oi], bottom=bottom, label=name)
        bottom += rel[oi]
    plt.xlabel("Query peptide length")
    plt.ylabel("Proportion")
    plt.title(title + " (relative)")
    plt.legend()
    safe_savefig(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}")

def plot_score_vs_entropy(
    df: pd.DataFrame,
    *,
    score_col: str,
    entropy_col: str,
    outbase: pathlib.Path,
    title: str,
    score_label: str,
    score_edges: np.ndarray,
    entropy_edges: Optional[np.ndarray] = None,
) -> None:
    if df.empty or score_col not in df or entropy_col not in df:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    sub = df[[score_col, entropy_col]].copy()
    sub = sub[np.isfinite(sub[score_col]) & np.isfinite(sub[entropy_col])]
    if sub.empty:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    if entropy_edges is None:
        emax = max(0.5, float(np.nanmax(sub[entropy_col])))
        entropy_edges = np.linspace(0.0, emax + 1e-9, 41)
    hist, _, _ = np.histogram2d(
        sub[entropy_col].to_numpy(dtype=float),
        sub[score_col].to_numpy(dtype=float),
        bins=[entropy_edges, score_edges],
    )
    plt.figure(figsize=(9, 6))
    mesh = plt.pcolormesh(score_edges, entropy_edges, hist, shading="auto")
    cb = plt.colorbar(mesh)
    cb.set_label("Count")
    plt.xlabel(score_label)
    plt.ylabel("Nearest-decoy Shannon entropy (bits)")
    plt.title(title)
    safe_savefig(outbase.with_suffix(_plot_ext()))
    rel = hist / max(1.0, float(hist.sum()))
    plt.figure(figsize=(9, 6))
    mesh = plt.pcolormesh(score_edges, entropy_edges, rel, shading="auto")
    cb = plt.colorbar(mesh)
    cb.set_label("Proportion of targets")
    plt.xlabel(score_label)
    plt.ylabel("Nearest-decoy Shannon entropy (bits)")
    plt.title(title + " (relative)")
    safe_savefig(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}")

def plot_mean_score_by_length_and_entropy(
    df: pd.DataFrame,
    *,
    score_col: str,
    entropy_col: str,
    outpath: pathlib.Path,
    title: str,
    entropy_edges: Optional[np.ndarray] = None,
) -> None:
    if df.empty or score_col not in df or entropy_col not in df or "query_length" not in df:
        empty_plot(outpath, title)
        return
    sub = df[["query_length", score_col, entropy_col]].copy()
    sub = sub[np.isfinite(sub[score_col]) & np.isfinite(sub[entropy_col])]
    if sub.empty:
        empty_plot(outpath, title)
        return
    if entropy_edges is None:
        emax = max(0.5, float(np.nanmax(sub[entropy_col])))
        entropy_edges = np.linspace(0.0, emax + 1e-9, 31)
    length_edges = np.arange(sub["query_length"].min(), sub["query_length"].max() + 2) - 0.5
    sum_hist, _, _ = np.histogram2d(
        sub["query_length"].to_numpy(dtype=float),
        sub[entropy_col].to_numpy(dtype=float),
        bins=[length_edges, entropy_edges],
        weights=sub[score_col].to_numpy(dtype=float),
    )
    count_hist, _, _ = np.histogram2d(
        sub["query_length"].to_numpy(dtype=float),
        sub[entropy_col].to_numpy(dtype=float),
        bins=[length_edges, entropy_edges],
    )
    mean_hist = np.divide(sum_hist, count_hist, out=np.full_like(sum_hist, np.nan), where=count_hist > 0)
    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(entropy_edges, length_edges, mean_hist, shading="auto")
    cb = plt.colorbar(mesh)
    cb.set_label(f"Mean {score_col}")
    plt.xlabel("Nearest-decoy Shannon entropy (bits)")
    plt.ylabel("Query peptide length")
    plt.title(title)
    safe_savefig(outpath)

def plot_local_balance(nonsteal_df: pd.DataFrame, outdir: pathlib.Path) -> None:
    if nonsteal_df.empty:
        empty_plot(outdir / f"local_competitor_balance{_plot_ext()}", "Local target/decoy competitor balance")
        empty_plot(outdir / f"local_competitor_balance_relative{_plot_ext()}", "Local target/decoy competitor balance")
        empty_plot(outdir / f"local_decoy_fraction{_plot_ext()}", "Local decoy fraction")
        empty_plot(outdir / f"local_decoy_fraction_relative{_plot_ext()}", "Local decoy fraction")
        return
    ratio = np.log2((nonsteal_df["n_decoy_competitors"].to_numpy(dtype=float) + 1.0) /
                    (nonsteal_df["n_target_competitors"].to_numpy(dtype=float) + 1.0))
    plot_single_histograms(
        ratio,
        outdir / "local_competitor_balance",
        "Local target/decoy competitor balance",
        "log2((decoy competitors + 1)/(target competitors + 1))",
        bins=50,
    )
    plot_single_histograms(
        nonsteal_df["local_decoy_fraction"].to_numpy(dtype=float),
        outdir / "local_decoy_fraction",
        "Local decoy fraction inside target windows",
        "Decoy fraction in local precursor window",
        bins=np.linspace(0.0, 1.0, 41),
    )

def plot_winner_bars(df: pd.DataFrame, winner_col: str, outbase: pathlib.Path, title: str) -> None:
    if df.empty or winner_col not in df:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    order = ["target", "decoy", "tie", "missing"]
    counts = [(df[winner_col] == x).sum() for x in order]
    x = np.arange(len(order))
    plt.figure(figsize=(7, 5))
    plt.bar(x, counts)
    plt.xticks(x, order)
    plt.ylabel("Count")
    plt.title(title)
    safe_savefig(outbase.with_suffix(_plot_ext()))
    total = max(1, sum(counts))
    plt.figure(figsize=(7, 5))
    plt.bar(x, np.asarray(counts, dtype=float) / total)
    plt.xticks(x, order)
    plt.ylabel("Proportion")
    plt.title(title + " (relative)")
    safe_savefig(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}")

def plot_winner_by_length(df: pd.DataFrame, winner_col: str, outbase: pathlib.Path, title: str) -> None:
    if df.empty or winner_col not in df:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    order = ["target", "decoy", "tie", "missing"]
    grouped = df.groupby("query_length")[winner_col]
    lengths = sorted(grouped.groups)
    count_table = np.zeros((len(order), len(lengths)), dtype=float)
    for li, length in enumerate(lengths):
        vals = df.loc[df["query_length"] == length, winner_col].astype(str)
        for oi, name in enumerate(order):
            count_table[oi, li] = float((vals == name).sum())
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(lengths), dtype=float)
    for oi, name in enumerate(order):
        plt.bar(lengths, count_table[oi], bottom=bottom, label=name)
        bottom += count_table[oi]
    plt.xlabel("Query peptide length")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    safe_savefig(outbase.with_suffix(_plot_ext()))
    row_sum = count_table.sum(axis=0, keepdims=True)
    rel = np.divide(count_table, row_sum, out=np.zeros_like(count_table), where=row_sum > 0)
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(lengths), dtype=float)
    for oi, name in enumerate(order):
        plt.bar(lengths, rel[oi], bottom=bottom, label=name)
        bottom += rel[oi]
    plt.xlabel("Query peptide length")
    plt.ylabel("Proportion")
    plt.title(title + " (relative)")
    plt.legend()
    safe_savefig(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}")

def plot_target_win_rate_by_length(
    df: pd.DataFrame,
    winner_col: str,
    outpath: pathlib.Path,
    title: str,
    min_count: int,
) -> None:
    if df.empty or winner_col not in df:
        empty_plot(outpath, title)
        return
    rows = []
    for length, grp in df.groupby("query_length"):
        n = len(grp)
        if n < min_count:
            continue
        vals = grp[winner_col].astype(str)
        valid = vals[vals != "missing"]
        if len(valid) == 0:
            continue
        eff = float((valid == "target").sum()) + 0.5 * float((valid == "tie").sum())
        rate = eff / len(valid)
        lo, hi = wilson_interval(eff, len(valid))
        expected = float(np.nanmean(grp["expected_target_win_rate"].to_numpy(dtype=float)))
        rows.append({"query_length": length, "rate": rate, "lo": lo, "hi": hi, "expected": expected})
    rate_df = pd.DataFrame(rows).sort_values("query_length") if rows else pd.DataFrame()
    if rate_df.empty:
        empty_plot(outpath, title)
        return
    x = rate_df["query_length"].to_numpy(dtype=float)
    y = rate_df["rate"].to_numpy(dtype=float)
    lo = rate_df["lo"].to_numpy(dtype=float)
    hi = rate_df["hi"].to_numpy(dtype=float)
    exp = rate_df["expected"].to_numpy(dtype=float)
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, label="observed effective target-win rate")
    #plt.fill_between(x, lo, hi, alpha=0.2)
    plt.plot(x, exp, label="expected from local competitor counts")
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Query peptide length")
    plt.ylabel("Target-win rate")
    plt.title(title)
    plt.legend()
    safe_savefig(outpath)

def plot_target_win_rate_by_mass(
    df: pd.DataFrame,
    winner_col: str,
    outpath: pathlib.Path,
    title: str,
    min_count: int,
    n_bins: int,
) -> None:
    if df.empty or winner_col not in df:
        empty_plot(outpath, title)
        return
    temp = df[["query_mass", "expected_target_win_rate", winner_col]].copy()
    temp = temp[np.isfinite(temp["query_mass"])]
    if temp.empty:
        empty_plot(outpath, title)
        return
    edges = np.unique(np.quantile(temp["query_mass"], np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        empty_plot(outpath, title)
        return
    temp["mass_bin"] = pd.cut(temp["query_mass"], bins=edges, include_lowest=True, duplicates="drop")
    rows = []
    for _, grp in temp.groupby("mass_bin", observed=False):
        n = len(grp)
        if n < min_count:
            continue
        vals = grp[winner_col].astype(str)
        valid = vals[vals != "missing"]
        if len(valid) == 0:
            continue
        eff = float((valid == "target").sum()) + 0.5 * float((valid == "tie").sum())
        rate = eff / len(valid)
        lo, hi = wilson_interval(eff, len(valid))
        rows.append(
            {
                "mass_center": float(np.nanmedian(grp["query_mass"])),
                "rate": rate,
                "lo": lo,
                "hi": hi,
                "expected": float(np.nanmean(grp["expected_target_win_rate"])),
            }
        )
    rate_df = pd.DataFrame(rows).sort_values("mass_center") if rows else pd.DataFrame()
    if rate_df.empty:
        empty_plot(outpath, title)
        return
    x = rate_df["mass_center"].to_numpy(dtype=float)
    y = rate_df["rate"].to_numpy(dtype=float)
    lo = rate_df["lo"].to_numpy(dtype=float)
    hi = rate_df["hi"].to_numpy(dtype=float)
    exp = rate_df["expected"].to_numpy(dtype=float)
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, label="observed effective target-win rate")
    #plt.fill_between(x, lo, hi, alpha=0.2)
    plt.plot(x, exp, label="expected from local competitor counts")
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Query peptide mass (Da)")
    plt.ylabel("Target-win rate")
    plt.title(title)
    plt.legend()
    safe_savefig(outpath)

def plot_score_overlays(df: pd.DataFrame, target_col: str, decoy_col: str, outbase: pathlib.Path, title: str, xlabel: str, bins=50) -> None:
    if df.empty:
        empty_plot(outbase.with_suffix(_plot_ext()), title)
        empty_plot(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}", title + " (relative)")
        return
    t = finite_values(df[target_col])
    d = finite_values(df[decoy_col])
    plt.figure(figsize=(8, 5))
    if t.size:
        plt.hist(t, bins=bins, alpha=0.6, label="best false target")
    if d.size:
        plt.hist(d, bins=bins, alpha=0.6, label="best decoy")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    safe_savefig(outbase.with_suffix(_plot_ext()))
    plt.figure(figsize=(8, 5))
    if t.size:
        plt.hist(t, bins=bins, alpha=0.6, label="best false target", weights=np.ones_like(t) / max(1, t.size))
    if d.size:
        plt.hist(d, bins=bins, alpha=0.6, label="best decoy", weights=np.ones_like(d) / max(1, d.size))
    plt.xlabel(xlabel)
    plt.ylabel("Proportion")
    plt.title(title + " (relative)")
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    safe_savefig(outbase.parent / f"{outbase.stem}_relative{_plot_ext()}")

def write_suspicious_table(nonsteal_df: pd.DataFrame, path: pathlib.Path, top_n: int) -> None:
    if nonsteal_df.empty:
        pd.DataFrame().to_csv(path, sep="	", index=False)
        return
    df = nonsteal_df.copy()
    sf = df["best_decoy_shared_fraction"].to_numpy(dtype=float)
    cos = df["best_decoy_cosine"].to_numpy(dtype=float)
    sf_rank = pd.Series(sf).rank(method="average", pct=True).to_numpy(dtype=float)
    cos_rank = pd.Series(cos).rank(method="average", pct=True).to_numpy(dtype=float)
    df["suspicious_rank_shared_fraction"] = sf_rank
    df["suspicious_rank_cosine"] = cos_rank
    df["suspicious_score"] = 0.5 * sf_rank + 0.5 * cos_rank
    df = df.sort_values(
        ["suspicious_score", "best_decoy_cosine", "best_decoy_shared_fraction", "best_decoy_shared"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    keep = [
        "query_sequence",
        "query_length",
        "query_mass",
        "query_source_class",
        "query_entropy",
        "query_protein_roots",
        "n_target_competitors",
        "n_decoy_competitors",
        "local_decoy_fraction",
        "best_decoy_sequence_shared",
        "best_decoy_mass_delta_shared",
        "best_decoy_same_protein_cross_version_shared",
        "best_decoy_same_protein_cross_version_shared_category",
        "best_decoy_entropy_shared",
        "best_decoy_shared",
        "best_decoy_shared_fraction",
        "best_decoy_sequence_cosine",
        "best_decoy_mass_delta_cosine",
        "best_decoy_same_protein_cross_version_cosine",
        "best_decoy_same_protein_cross_version_cosine_category",
        "best_decoy_entropy_cosine",
        "best_decoy_cosine",
        "best_decoy_cosine_distance",
        "suspicious_rank_shared_fraction",
        "suspicious_rank_cosine",
        "suspicious_score",
    ]
    df[keep].head(top_n).to_csv(path, sep="	", index=False)

def summarize_outputs(
    peptide_df: pd.DataFrame,
    null_df: pd.DataFrame,
    nonsteal_df: pd.DataFrame,
    random_null_df: pd.DataFrame,
    meta: dict,
) -> dict:
    out = dict(meta)
    if not peptide_df.empty:
        out["n_target_only_sequences"] = int((peptide_df["source_class"] == "target_only").sum())
        out["n_decoy_only_sequences"] = int((peptide_df["source_class"] == "decoy_only").sum())
        out["n_overlap_promoted_sequences"] = int((peptide_df["source_class"] == "overlap_promoted_to_target").sum())
    if not nonsteal_df.empty:
        for col in [
            "best_decoy_shared",
            "best_decoy_shared_fraction",
            "best_decoy_cosine",
            "best_decoy_cosine_distance",
            "best_decoy_entropy_shared",
            "best_decoy_entropy_cosine",
        ]:
            vals = finite_values(nonsteal_df[col]) if col in nonsteal_df else np.array([], dtype=float)
            if vals.size:
                out[f"{col}_median"] = float(np.median(vals))
                out[f"{col}_p95"] = float(np.quantile(vals, 0.95))
        if "best_decoy_same_protein_cross_version_shared_category" in nonsteal_df:
            cats = nonsteal_df["best_decoy_same_protein_cross_version_shared_category"].astype(str)
            valid = cats[cats != "missing"]
            if len(valid):
                out["nearest_decoy_same_protein_cross_version_rate_shared"] = float((valid == "same_protein_cross_version").mean())
        if "best_decoy_same_protein_cross_version_cosine_category" in nonsteal_df:
            cats = nonsteal_df["best_decoy_same_protein_cross_version_cosine_category"].astype(str)
            valid = cats[cats != "missing"]
            if len(valid):
                out["nearest_decoy_same_protein_cross_version_rate_cosine"] = float((valid == "same_protein_cross_version").mean())
    if not null_df.empty:
        for metric in ["shared", "cosine"]:
            winner_col = f"winner_{metric}"
            valid = null_df[winner_col].astype(str)
            valid = valid[valid != "missing"]
            if len(valid):
                eff = float((valid == "target").sum()) + 0.5 * float((valid == "tie").sum())
                rate = eff / len(valid)
                lo, hi = wilson_interval(eff, len(valid))
                out[f"null_{metric}_effective_target_win_rate"] = rate
                out[f"null_{metric}_effective_target_win_rate_ci_low"] = lo
                out[f"null_{metric}_effective_target_win_rate_ci_high"] = hi
                out[f"null_{metric}_expected_target_win_rate_mean"] = float(np.nanmean(null_df["expected_target_win_rate"].to_numpy(dtype=float)))
            target_col = f"best_false_target_{metric}"
            decoy_col = f"best_decoy_{metric}"
            out[f"null_{metric}_ks_false_target_vs_decoy"] = ks_2samp_basic(
                null_df[target_col].to_numpy(dtype=float),
                null_df[decoy_col].to_numpy(dtype=float),
            )
            for qlab, prefix in [("target", "best_decoy"), ("decoy", "best_false_target")]:
                cat_col = f"{prefix}_same_protein_cross_version_{metric}_category"
                sub = null_df[null_df["query_label"] == qlab]
                if cat_col in sub and not sub.empty:
                    cats = sub[cat_col].astype(str)
                    valid = cats[cats != "missing"]
                    if len(valid):
                        out[f"{qlab}_queries_same_protein_cross_version_rate_{metric}"] = float((valid == "same_protein_cross_version").mean())
    if not random_null_df.empty:
        for metric in ["shared", "cosine"]:
            winner_col = f"winner_{metric}"
            valid = random_null_df[winner_col].astype(str)
            valid = valid[valid != "missing"]
            if len(valid):
                eff = float((valid == "target").sum()) + 0.5 * float((valid == "tie").sum())
                out[f"random_null_{metric}_effective_target_win_rate"] = eff / len(valid)
                out[f"random_null_{metric}_expected_target_win_rate_mean"] = float(np.nanmean(random_null_df["expected_target_win_rate"].to_numpy(dtype=float)))
    return out

def write_chart_guide(outdir: pathlib.Path) -> None:
    text = textwrap.dedent(
        """
        # Decoy diagnostics chart guide
        ## Overlap handling
        If an exact peptide sequence appears in both target and decoy protein digests,
        it is **promoted to target** for all competition analyses. This mirrors the usual
        search-engine convention that an exact target/decoy tie at the sequence level
        should be treated as target, not as a separate decoy competitor.
        The overlap list is written to:
        - `target_decoy_sequence_overlaps.tsv`
        - `exact_target_decoy_sequence_overlaps.txt`
        ## Absolute vs relative plots
        For most histogram / bar / heatmap plot families the script writes two versions:
        - `*.png`: absolute counts
        - `*_relative.png`: normalized proportions
        Relative plots are especially important when target and effective decoy counts are unequal.
        ## Main plot families
        - `length_distribution*.png`: target/decoy peptide length distributions
        - `mass_distribution*.png`: target/decoy peptide mass distributions
        - `mass_bin_counts*.png`: target/decoy counts/proportions by precursor-mass bin
        - `mass_bin_decoy_target_ratio.png`: decoy/target ratio by precursor-mass bin
        - `mass_length_balance_heatmap*.png`: absolute peptide density and relative target/decoy balance by mass and length
        - `local_competitor_balance*.png`: local precursor-window target/decoy competitor imbalance
        - `local_decoy_fraction*.png`: decoy fraction in local target windows
        ## Non-stealability and suspicious-neighborhood plots
        These plots use target queries only and summarize the best effective-decoy competitor in each precursor window.
        - `nonsteal_best_decoy_shared*.png`: best shared theoretical b/y-ion count
        - `nonsteal_best_decoy_shared_fraction*.png`: shared-ion count normalized by query fragment count
        - `nonsteal_best_decoy_cosine*.png`: best spectral cosine
        - `nonsteal_best_decoy_cosine_distance*.png`: best spectral cosine distance (= 1 - cosine)
        - `nearest_decoy_cosine_distance_distribution*.png`: the distance to the nearest decoy across all targets
        - `closest_decoy_*_vs_entropy*.png`: relationship between nearest-decoy score and nearest-decoy Shannon entropy
        - `closest_decoy_*_vs_entropy_by_length_mean.png`: mean nearest-decoy score as a function of target length and nearest-decoy entropy
        The `_by_length` variants are heatmaps:
        - absolute version = counts of targets in each (length, score) cell
        - relative version = row-normalized proportions within each peptide length
        A concerning suspicious pattern is:
        - a large mass of targets with very small nearest-decoy cosine distance
        - high nearest-decoy similarity concentrated among low-entropy decoys
        - strongest low-entropy effect in short peptides
        ## Same-protein cross-version diagnostics
        Using the `rev_` prefix convention, the script asks whether the closest opposite-label peptide
        comes from the same underlying protein but in the other database version.
        - `same_protein_cross_version_*_overall_*_queries*.png`: global counts/proportions of same-protein vs other-protein vs missing
        - `same_protein_cross_version_*_by_length_*_queries*.png`: the same breakdown by peptide length
        For target queries this checks whether the best decoy comes from the reversed version of the same target protein.
        For decoy queries this checks whether the best false target comes from the original target protein.
        ## Null diagnostics
        These compare the best false-target score with the best decoy score.
        - `null_*_delta_*_queries*.png`: delta score histograms
        - `null_*_score_overlay_*_queries*.png`: false-target vs decoy score overlays
        - `null_*_winner_overall_*_queries*.png`: overall winner counts / proportions
        - `null_*_winner_by_length_*_queries*.png`: winner composition by length
        - `null_*_target_win_rate_by_length_*_queries.png`: observed effective target-win rate vs the expected rate from local competitor counts
        - `null_*_target_win_rate_by_mass_*_queries.png`: same idea but stratified by mass
        A good null pattern is:
        - false-target and decoy score distributions overlap
        - observed effective target-win rate roughly tracks the expected rate from local candidate counts
        ## Suspicious target table
        `tables/top_suspicious_targets.tsv` ranks target queries by a combined score using:
        - best decoy shared-ion fraction
        - best decoy cosine
        It also records nearest-decoy sequence IDs, mass deltas, same-protein flags, and Shannon entropy.
        """
    ).strip() + "\n"
    (outdir / "CHART_GUIDE.md").write_text(text)

def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Digest a target/decoy FASTA and generate decoy diagnostics.",
    )
    p.add_argument("--fasta", required=False, default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--rerun-random-null", action="store_true",
                   help="Re-run only the random-null sweep and plots. "
                        "Loads peptides, null queries, and spectra from the "
                        "previous run in --outdir. --fasta and spectra args "
                        "are read from run_config.json if not given.")
    p.add_argument("--decoy-tag", default="rev_")
    p.add_argument("--missed-cleavages", type=int, default=2)
    p.add_argument("--min-length", type=int, default=7)
    p.add_argument("--max-length", type=int, default=30)
    p.add_argument("--mass-window", type=float, default=0.5, help="Competition window on neutral monoisotopic mass (Da)")
    p.add_argument("--fragment-tol", type=float, default=0.02, help="Absolute fragment m/z tolerance (Da)")
    p.add_argument("--fragment-charges", type=parse_int_list, default=[1], help="Comma-separated fragment charges for theoretical b/y ions")
    p.add_argument("--sample-targets", type=int, default=0, help="0 = all")
    p.add_argument("--sample-decoys", type=int, default=0, help="0 = all")
    p.add_argument("--random-null-queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--min-count-per-length-plot", type=int, default=20)
    p.add_argument("--mass-winrate-bins", type=int, default=20)
    p.add_argument("--top-suspicious", type=int, default=200)
    perf = p.add_argument_group("Performance")
    perf.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    perf.add_argument("--parallel-backend", choices=["auto", "process", "thread", "none"], default="auto")
    perf.add_argument("--parallel-batch-size", type=int, default=512)
    pred = p.add_argument_group("Predicted spectra")
    pred.add_argument("--predicted-spectra", default=None, help="Optional long TSV/CSV with peptide/sequence, mz, intensity columns")
    pred.add_argument("--precursor-charge", type=int, default=2)
    pred.add_argument("--collision-energy", type=float, default=27.0)
    pred.add_argument("--koina-url", default=None)
    pred.add_argument("--koina-model", default="Prosit_2019_intensity")
    pred.add_argument("--koina-batch-size", type=int, default=128)
    pred.add_argument("--pred-top-k", type=int, default=50)
    pred.add_argument("--pred-min-intensity", type=float, default=1e-6)
    return p

def _load_spectra_for_rerun(args, tables_dir: pathlib.Path) -> Dict[str, Spectrum]:
    """Load spectra from cache or original source for --rerun-random-null."""
    spectra: Dict[str, Spectrum] = {}
    cache_path = tables_dir / "spectra_cache.tsv"
    if cache_path.exists():
        log("Loading spectra from cache ...")
        spectra.update(load_spectra_cache(cache_path))
        log(f"Loaded {len(spectra)} spectra from cache")
    if args.predicted_spectra:
        log("Loading predicted spectra table ...")
        spectra.update(
            load_predicted_spectra(
                args.predicted_spectra,
                precursor_charge=args.precursor_charge,
                collision_energy=args.collision_energy,
                top_k=args.pred_top_k,
                min_intensity=args.pred_min_intensity,
            )
        )
        log(f"Loaded predicted spectra for {len(spectra)} peptides")
    return spectra

def main() -> None:
    args = build_arg_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    plots_dir = ensure_dir(outdir / "plots")
    tables_dir = ensure_dir(outdir / "tables")
    plan = resolve_parallel_plan(args.workers, args.parallel_backend, args.parallel_batch_size)
    log(f"Parallel plan: backend={plan.backend}, workers={plan.workers}, batch_size={plan.batch_size}")

    if args.rerun_random_null:
        # ---------- fast path: reload everything except random-null ----------
        config_path = outdir / "run_config.json"
        if not config_path.exists():
            raise SystemExit(f"Cannot rerun: {config_path} not found")
        with open(config_path) as handle:
            prev = json.load(handle)
        # Inherit args that default to None from the previous run.
        for key in ["fasta", "predicted_spectra", "koina_url"]:
            if getattr(args, key, None) is None and key in prev:
                setattr(args, key, prev[key])
        if not args.fasta:
            raise SystemExit("--fasta is required (not found in run_config.json either)")
        log("Rerun mode: loading peptides from saved table ...")
        peptide_df = pd.read_csv(tables_dir / "peptides.tsv", sep="\t")
        if peptide_df.empty:
            raise SystemExit("Saved peptides.tsv is empty.")
        meta = {
            "n_effective_targets": int((peptide_df["effective_label"] == "target").sum()),
            "n_effective_decoys": int((peptide_df["effective_label"] == "decoy").sum()),
            "n_target_decoy_overlaps": int(peptide_df["is_target_decoy_overlap"].sum()),
        }
        log("Rerun mode: loading null queries from saved table ...")
        null_df = pd.read_csv(tables_dir / "null_queries.tsv", sep="\t")
        nonsteal_df = derive_nonstealability(null_df)
        log("Rerun mode: rebuilding peptide entries ...")
        spectra = _load_spectra_for_rerun(args, tables_dir)
        target_entries, decoy_entries, entry_meta = build_entries(
            peptide_df,
            fragment_charges=args.fragment_charges,
            spectra=spectra,
            pred_top_k=args.pred_top_k,
        )
        meta.update(entry_meta)
        log(f"Running random null sweeps ({args.random_null_queries} queries) ...")
        random_null_df = simulate_random_null_queries(
            target_entries,
            decoy_entries,
            mass_window=args.mass_window,
            fragment_tol=args.fragment_tol,
            n_queries=args.random_null_queries,
            seed=args.seed + 11,
            plan=plan,
        )
        if not random_null_df.empty:
            random_null_df.to_csv(tables_dir / "random_null_queries.tsv", sep="\t", index=False)
        # Update run_config with the new random-null count.
        with open(outdir / "run_config.json", "wt") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)
    else:
        # ---------- full run ----------
        if not args.fasta:
            raise SystemExit("--fasta is required")
        with open(outdir / "run_config.json", "wt") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)
        peptide_df, meta = build_peptide_table(
            args.fasta,
            decoy_tag=args.decoy_tag,
            missed_cleavages=args.missed_cleavages,
            min_length=args.min_length,
            max_length=args.max_length,
        )
        if peptide_df.empty:
            raise SystemExit("No peptides were produced from digestion.")
        peptide_df = peptide_df.sort_values(["effective_label", "mass", "sequence"]).reset_index(drop=True)
        overlap_meta = write_overlap_reports(peptide_df, outdir)
        meta.update(overlap_meta)
        peptide_df.to_csv(tables_dir / "peptides.tsv", sep="\t", index=False)
        log(f"Effective library: {meta['n_effective_targets']} targets, {meta['n_effective_decoys']} decoys, {meta['n_target_decoy_overlaps']} exact overlaps promoted to target")
        spectra: Dict[str, Spectrum] = {}
        if args.predicted_spectra:
            log("Loading predicted spectra table ...")
            spectra.update(
                load_predicted_spectra(
                    args.predicted_spectra,
                    precursor_charge=args.precursor_charge,
                    collision_energy=args.collision_energy,
                    top_k=args.pred_top_k,
                    min_intensity=args.pred_min_intensity,
                )
            )
            log(f"Loaded predicted spectra for {len(spectra)} peptides")
        if args.koina_url:
            needed = sorted(set(peptide_df["sequence"]) - set(spectra))
            if needed:
                log(f"Requesting Koina predictions for {len(needed)} peptides ...")
                koina_spectra = fetch_koina_predictions(
                    needed,
                    url=args.koina_url,
                    model_name=args.koina_model,
                    precursor_charge=args.precursor_charge,
                    collision_energy=args.collision_energy,
                    batch_size=args.koina_batch_size,
                    top_k=args.pred_top_k,
                    min_intensity=args.pred_min_intensity,
                )
                spectra.update(koina_spectra)
                write_spectra_cache(tables_dir / "spectra_cache.tsv", spectra)
        target_entries, decoy_entries, entry_meta = build_entries(
            peptide_df,
            fragment_charges=args.fragment_charges,
            spectra=spectra,
            pred_top_k=args.pred_top_k,
        )
        meta.update(entry_meta)
        log("Running target/decoy competition sweeps ...")
        null_df = compute_null_queries(
            target_entries,
            decoy_entries,
            mass_window=args.mass_window,
            fragment_tol=args.fragment_tol,
            sample_targets=args.sample_targets,
            sample_decoys=args.sample_decoys,
            seed=args.seed,
            plan=plan,
        )
        null_df.to_csv(tables_dir / "null_queries.tsv", sep="\t", index=False)
        nonsteal_df = derive_nonstealability(null_df)
        nonsteal_df.to_csv(tables_dir / "nonstealability.tsv", sep="\t", index=False)
        write_suspicious_table(nonsteal_df, tables_dir / "top_suspicious_targets.tsv", args.top_suspicious)
        log("Running random null sweeps ...")
        random_null_df = simulate_random_null_queries(
            target_entries,
            decoy_entries,
            mass_window=args.mass_window,
            fragment_tol=args.fragment_tol,
            n_queries=args.random_null_queries,
            seed=args.seed + 11,
            plan=plan,
        )
        if not random_null_df.empty:
            random_null_df.to_csv(tables_dir / "random_null_queries.tsv", sep="\t", index=False)
    log("Writing plots ...")
    plot_two_label_histograms(
        peptide_df, "length", "effective_label", plots_dir / "length_distribution",
        "Peptide length distribution", "Peptide length", bins=np.arange(peptide_df["length"].min(), peptide_df["length"].max() + 2) - 0.5
    )
    plot_two_label_histograms(
        peptide_df, "mass", "effective_label", plots_dir / "mass_distribution",
        "Peptide mass distribution", "Neutral monoisotopic peptide mass (Da)", bins=60
    )
    plot_mass_bin_counts(peptide_df, plots_dir)
    plot_mass_length_balance_heatmap(peptide_df, plots_dir)
    plot_local_balance(nonsteal_df, plots_dir)
    # Non-stealability histograms and by-length heatmaps.
    shared_vals = finite_values(nonsteal_df["best_decoy_shared"]) if not nonsteal_df.empty else np.array([], dtype=float)
    max_shared = int(np.max(shared_vals)) if shared_vals.size else 0
    shared_edges = np.arange(-0.5, max(5, max_shared) + 1.5, 1.0)
    plot_single_histograms(
        nonsteal_df["best_decoy_shared"].to_numpy(dtype=float),
        plots_dir / "nonsteal_best_decoy_shared",
        "Non-stealability: best decoy shared-ion count per target",
        "Best decoy shared b/y-ion count",
        bins=shared_edges,
    )
    plot_1d_heatmap_by_length(
        nonsteal_df,
        metric_col="best_decoy_shared",
        outbase=plots_dir / "nonsteal_best_decoy_shared_by_length",
        title="Non-stealability: best decoy shared-ion count by target length",
        xlabel="Best decoy shared b/y-ion count",
        metric_edges=shared_edges,
    )
    plot_single_histograms(
        nonsteal_df["best_decoy_shared_fraction"].to_numpy(dtype=float),
        plots_dir / "nonsteal_best_decoy_shared_fraction",
        "Non-stealability: best decoy shared-ion fraction per target",
        "Best decoy shared-ion fraction",
        bins=np.linspace(0.0, 1.0, 41),
    )
    plot_1d_heatmap_by_length(
        nonsteal_df,
        metric_col="best_decoy_shared_fraction",
        outbase=plots_dir / "nonsteal_best_decoy_shared_fraction_by_length",
        title="Non-stealability: best decoy shared-ion fraction by target length",
        xlabel="Best decoy shared-ion fraction",
        metric_edges=np.linspace(0.0, 1.0, 41),
    )
    plot_single_histograms(
        nonsteal_df["best_decoy_cosine"].to_numpy(dtype=float),
        plots_dir / "nonsteal_best_decoy_cosine",
        "Non-stealability: best decoy cosine per target",
        "Best decoy cosine",
        bins=np.linspace(0.0, 1.0, 41),
    )
    plot_1d_heatmap_by_length(
        nonsteal_df,
        metric_col="best_decoy_cosine",
        outbase=plots_dir / "nonsteal_best_decoy_cosine_by_length",
        title="Non-stealability: best decoy cosine by target length",
        xlabel="Best decoy cosine",
        metric_edges=np.linspace(0.0, 1.0, 41),
    )
    plot_single_histograms(
        nonsteal_df["best_decoy_cosine_distance"].to_numpy(dtype=float),
        plots_dir / "nonsteal_best_decoy_cosine_distance",
        "Non-stealability: best decoy cosine distance per target",
        "Best decoy cosine distance (= 1 - cosine)",
        bins=np.linspace(0.0, 1.0, 41),
    )
    plot_1d_heatmap_by_length(
        nonsteal_df,
        metric_col="best_decoy_cosine_distance",
        outbase=plots_dir / "nonsteal_best_decoy_cosine_distance_by_length",
        title="Non-stealability: best decoy cosine distance by target length",
        xlabel="Best decoy cosine distance",
        metric_edges=np.linspace(0.0, 1.0, 41),
    )
    # Suspicious-neighborhood diagnostics.
    plot_single_histograms(
        nonsteal_df["best_decoy_cosine_distance"].to_numpy(dtype=float),
        plots_dir / "nearest_decoy_cosine_distance_distribution",
        "Suspicious diagnostic: distance to the nearest decoy across all targets",
        "Nearest-decoy cosine distance (= 1 - best decoy cosine)",
        bins=np.linspace(0.0, 1.0, 41),
    )
    plot_1d_heatmap_by_length(
        nonsteal_df,
        metric_col="best_decoy_cosine_distance",
        outbase=plots_dir / "nearest_decoy_cosine_distance_distribution_by_length",
        title="Suspicious diagnostic: nearest-decoy cosine distance by target length",
        xlabel="Nearest-decoy cosine distance",
        metric_edges=np.linspace(0.0, 1.0, 41),
    )
    entropy_candidates = np.concatenate([
        finite_values(nonsteal_df["best_decoy_entropy_shared"]) if "best_decoy_entropy_shared" in nonsteal_df else np.array([], dtype=float),
        finite_values(nonsteal_df["best_decoy_entropy_cosine"]) if "best_decoy_entropy_cosine" in nonsteal_df else np.array([], dtype=float),
    ])
    entropy_max = float(np.max(entropy_candidates)) if entropy_candidates.size else 4.0
    entropy_edges = np.linspace(0.0, max(1.0, entropy_max) + 1e-9, 31)
    plot_score_vs_entropy(
        nonsteal_df,
        score_col="best_decoy_shared_fraction",
        entropy_col="best_decoy_entropy_shared",
        outbase=plots_dir / "closest_decoy_shared_fraction_vs_entropy",
        title="Suspicious diagnostic: closest-decoy shared fraction vs nearest-decoy Shannon entropy",
        score_label="Closest-decoy shared-ion fraction",
        score_edges=np.linspace(0.0, 1.0, 41),
        entropy_edges=entropy_edges,
    )
    plot_score_vs_entropy(
        nonsteal_df,
        score_col="best_decoy_cosine",
        entropy_col="best_decoy_entropy_cosine",
        outbase=plots_dir / "closest_decoy_cosine_vs_entropy",
        title="Suspicious diagnostic: closest-decoy cosine vs nearest-decoy Shannon entropy",
        score_label="Closest-decoy cosine",
        score_edges=np.linspace(0.0, 1.0, 41),
        entropy_edges=entropy_edges,
    )
    plot_mean_score_by_length_and_entropy(
        nonsteal_df,
        score_col="best_decoy_shared_fraction",
        entropy_col="best_decoy_entropy_shared",
        outpath=plots_dir / f"closest_decoy_shared_fraction_vs_entropy_by_length_mean{_plot_ext()}",
        title="Suspicious diagnostic: mean closest-decoy shared fraction by target length and nearest-decoy entropy",
        entropy_edges=entropy_edges,
    )
    plot_mean_score_by_length_and_entropy(
        nonsteal_df,
        score_col="best_decoy_cosine",
        entropy_col="best_decoy_entropy_cosine",
        outpath=plots_dir / f"closest_decoy_cosine_vs_entropy_by_length_mean{_plot_ext()}",
        title="Suspicious diagnostic: mean closest-decoy cosine by target length and nearest-decoy entropy",
        entropy_edges=entropy_edges,
    )
    # Null diagnostics by query label.
    for qlab in ["target", "decoy"]:
        sub = null_df[null_df["query_label"] == qlab].copy()
        if sub.empty:
            continue
        shared_delta = sub["delta_shared_target_minus_decoy"].to_numpy(dtype=float)
        finite_delta = finite_values(shared_delta)
        delta_bins = np.arange(np.floor(finite_delta.min()) - 0.5, np.ceil(finite_delta.max()) + 1.5, 1.0) if finite_delta.size else 25
        plot_single_histograms(
            shared_delta,
            plots_dir / f"null_shared_delta_{qlab}_queries",
            f"Null diagnostic ({qlab} queries): false-target minus decoy shared-ion score",
            "best false target shared - best decoy shared",
            bins=delta_bins,
        )
        plot_score_overlays(
            sub, "best_false_target_shared", "best_decoy_shared",
            plots_dir / f"null_shared_score_overlay_{qlab}_queries",
            f"Null diagnostic ({qlab} queries): false-target vs decoy shared-ion scores",
            "Score",
            bins=delta_bins if isinstance(delta_bins, np.ndarray) else 50,
        )
        plot_winner_bars(
            sub, "winner_shared",
            plots_dir / f"null_shared_winner_overall_{qlab}_queries",
            f"Null diagnostic ({qlab} queries): winner composition (shared ions)",
        )
        plot_winner_by_length(
            sub, "winner_shared",
            plots_dir / f"null_shared_winner_by_length_{qlab}_queries",
            f"Null diagnostic ({qlab} queries): winner composition by length (shared ions)",
        )
        plot_target_win_rate_by_length(
            sub, "winner_shared",
            plots_dir / f"null_shared_target_win_rate_by_length_{qlab}_queries{_plot_ext()}",
            f"Null diagnostic ({qlab} queries): target-win rate by length (shared ions)",
            min_count=args.min_count_per_length_plot,
        )
        plot_target_win_rate_by_mass(
            sub, "winner_shared",
            plots_dir / f"null_shared_target_win_rate_by_mass_{qlab}_queries{_plot_ext()}",
            f"Null diagnostic ({qlab} queries): target-win rate by mass (shared ions)",
            min_count=args.min_count_per_length_plot,
            n_bins=args.mass_winrate_bins,
        )
        sameprot_shared_col = "best_decoy_same_protein_cross_version_shared_category" if qlab == "target" else "best_false_target_same_protein_cross_version_shared_category"
        plot_category_bars(
            sub,
            sameprot_shared_col,
            plots_dir / f"same_protein_cross_version_shared_overall_{qlab}_queries",
            f"Same-protein cross-version frequency ({qlab} queries, shared ions)",
            order=["same_protein_cross_version", "other_protein", "missing"],
        )
        plot_category_by_length(
            sub,
            sameprot_shared_col,
            plots_dir / f"same_protein_cross_version_shared_by_length_{qlab}_queries",
            f"Same-protein cross-version frequency by length ({qlab} queries, shared ions)",
            order=["same_protein_cross_version", "other_protein", "missing"],
        )
        plot_single_histograms(
            sub["delta_cosine_target_minus_decoy"].to_numpy(dtype=float),
            plots_dir / f"null_cosine_delta_{qlab}_queries",
            f"Null diagnostic ({qlab} queries): false-target minus decoy cosine",
            "best false target cosine - best decoy cosine",
            bins=np.linspace(-1.0, 1.0, 41),
        )
        plot_score_overlays(
            sub, "best_false_target_cosine", "best_decoy_cosine",
            plots_dir / f"null_cosine_score_overlay_{qlab}_queries",
            f"Null diagnostic ({qlab} queries): false-target vs decoy cosine",
            "Cosine",
            bins=np.linspace(0.0, 1.0, 41),
        )
        plot_winner_bars(
            sub, "winner_cosine",
            plots_dir / f"null_cosine_winner_overall_{qlab}_queries",
            f"Null diagnostic ({qlab} queries): winner composition (cosine)",
        )
        plot_winner_by_length(
            sub, "winner_cosine",
            plots_dir / f"null_cosine_winner_by_length_{qlab}_queries",
            f"Null diagnostic ({qlab} queries): winner composition by length (cosine)",
        )
        plot_target_win_rate_by_length(
            sub, "winner_cosine",
            plots_dir / f"null_cosine_target_win_rate_by_length_{qlab}_queries{_plot_ext()}",
            f"Null diagnostic ({qlab} queries): target-win rate by length (cosine)",
            min_count=args.min_count_per_length_plot,
        )
        plot_target_win_rate_by_mass(
            sub, "winner_cosine",
            plots_dir / f"null_cosine_target_win_rate_by_mass_{qlab}_queries{_plot_ext()}",
            f"Null diagnostic ({qlab} queries): target-win rate by mass (cosine)",
            min_count=args.min_count_per_length_plot,
            n_bins=args.mass_winrate_bins,
        )
        sameprot_cos_col = "best_decoy_same_protein_cross_version_cosine_category" if qlab == "target" else "best_false_target_same_protein_cross_version_cosine_category"
        plot_category_bars(
            sub,
            sameprot_cos_col,
            plots_dir / f"same_protein_cross_version_cosine_overall_{qlab}_queries",
            f"Same-protein cross-version frequency ({qlab} queries, cosine)",
            order=["same_protein_cross_version", "other_protein", "missing"],
        )
        plot_category_by_length(
            sub,
            sameprot_cos_col,
            plots_dir / f"same_protein_cross_version_cosine_by_length_{qlab}_queries",
            f"Same-protein cross-version frequency by length ({qlab} queries, cosine)",
            order=["same_protein_cross_version", "other_protein", "missing"],
        )
    if not random_null_df.empty:
        plot_winner_bars(
            random_null_df, "winner_shared",
            plots_dir / "random_null_shared_winner_overall",
            "Random null queries: winner composition (shared ions)",
        )
        plot_winner_bars(
            random_null_df, "winner_cosine",
            plots_dir / "random_null_cosine_winner_overall",
            "Random null queries: winner composition (cosine)",
        )
        plot_score_overlays(
            random_null_df, "best_target_shared", "best_decoy_shared",
            plots_dir / "random_null_shared_best_score_distributions",
            "Random null queries: best target vs best decoy shared-ion scores",
            "Score",
        )
        plot_score_overlays(
            random_null_df, "best_target_cosine", "best_decoy_cosine",
            plots_dir / "random_null_cosine_best_score_distributions",
            "Random null queries: best target vs best decoy cosine scores",
            "Cosine",
            bins=np.linspace(0.0, 1.0, 41),
        )
    write_chart_guide(outdir)
    summary = summarize_outputs(peptide_df, null_df, nonsteal_df, random_null_df, meta)
    with open(outdir / "summary.json", "wt") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(outdir / "README.txt", "wt") as handle:
        handle.write(
            textwrap.dedent(
                f"""
                Decoy diagnostics output
                =======================
                Effective overlap policy
                ------------------------
                Exact peptide sequences present in both target and decoy digests are promoted to target.
                They are excluded from the effective decoy pool used in all competition analyses.
                Tables
                ------
                {tables_dir}
                Plots
                -----
                {plots_dir}
                Key tables
                ----------
                peptides.tsv
                    Unique peptide sequences with raw target/decoy occurrence counts and effective label.
                target_decoy_sequence_overlaps.tsv
                    Exact target/decoy overlaps promoted to target.
                null_queries.tsv
                    Query-vs-library null diagnostics for target and decoy queries.
                nonstealability.tsv
                    Target-query subset used for non-stealability diagnostics.
                top_suspicious_targets.tsv
                    Highest-risk target neighborhoods ranked by shared-ion fraction and cosine.
                random_null_queries.tsv
                    Synthetic random-null queries (only if enabled).
                Plot naming
                -----------
                Most plot families come in two forms:
                * *.png            -> absolute counts
                * *_relative.png   -> normalized proportions
                Worker plan
                -----------
                backend={plan.backend}
                workers={plan.workers}
                batch_size={plan.batch_size}
                Notes
                -----
                * precursor mass window: +/- {args.mass_window} Da
                * fragment tolerance: {args.fragment_tol} Da
                * fragment charges: {args.fragment_charges}
                * spectra: predicted/Koina when available, otherwise theoretical stick spectra
                """
            ).strip() + "\n"
        )
    log("Done.")
if __name__ == "__main__":
    main()