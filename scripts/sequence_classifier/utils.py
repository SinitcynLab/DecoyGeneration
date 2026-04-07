from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decoy_generators.diann_generator import DiannGenerator
from src.decoy_generators.reverse_generator import ReverseGenerator
from src.decoy_generators.shuffle_generator import ShuffleGenerator
from src.io.fasta import FastaRecord, read_fasta_file
from src.io.peptide_processor import PeptideProcessor


DEFAULT_SPECIAL_AMINO_ACIDS = ["R", "K"]
DEFAULT_LENGTH_RANGE = (9, 40)
CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
EXPERIMENT_RE = re.compile(r"^(?P<row>\d+)M\.(?P<variant>.+)$")
VARIANT_TO_COLUMN = {
    "c_term": "c",
    "n_term": "n",
    "n_c_term": "n_c",
    "p01": "p_10",
    "p02": "p_20",
    "p03": "p_30",
}


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    header: str
    sequence: str


def _header_token(header: str) -> str:
    return header.split(maxsplit=1)[0]


def protein_id_from_header(header: str) -> str:
    parts = _header_token(header).split("|")
    return parts[1] if len(parts) >= 2 else parts[0]


def is_contaminant_header(header: str) -> bool:
    token = _header_token(header).lower()
    return token.startswith("contam") or token.startswith("rev_contam")


def is_decoy_header(header: str) -> bool:
    return any(part.startswith("rev_") for part in _header_token(header).split("|"))


def has_only_canonical_amino_acids(sequence: str) -> bool:
    return all(amino_acid in CANONICAL_AMINO_ACIDS for amino_acid in sequence)


def read_protein_records(
    fasta_path: str,
    canonical_only: bool = True,
) -> list[ProteinRecord]:
    return [
        ProteinRecord(
            protein_id=protein_id_from_header(record.head),
            header=record.head,
            sequence=record.sequence,
        )
        for record in read_fasta_file(fasta_path)
        if not canonical_only or has_only_canonical_amino_acids(record.sequence)
    ]


def split_target_and_decoy_proteins(fasta_path: str) -> tuple[list[ProteinRecord], list[ProteinRecord]]:
    proteins = [protein for protein in read_protein_records(fasta_path) if not is_contaminant_header(protein.header)]
    return (
        [protein for protein in proteins if not is_decoy_header(protein.header)],
        [protein for protein in proteins if is_decoy_header(protein.header)],
    )


def _create_generator(method: str, seed: int):
    factories = {
        "shuffle": lambda: ShuffleGenerator(
            special_amino_acids=DEFAULT_SPECIAL_AMINO_ACIDS,
            random=Random(seed),
        ),
        "reverse": lambda: ReverseGenerator(special_amino_acids=DEFAULT_SPECIAL_AMINO_ACIDS),
        "diann": lambda: DiannGenerator(special_amino_acids=DEFAULT_SPECIAL_AMINO_ACIDS),
    }
    try:
        return factories[method]()
    except KeyError as exc:
        raise ValueError(f"Unknown decoy generator method: {method}") from exc


def _protein_records_from_fasta_records(records: Iterable[FastaRecord]) -> list[ProteinRecord]:
    return [
        ProteinRecord(
            protein_id=protein_id_from_header(record.head),
            header=record.head,
            sequence=record.sequence,
        )
        for record in records
    ]


def _fasta_records_from_protein_records(proteins: Iterable[ProteinRecord]) -> list[FastaRecord]:
    return [FastaRecord(head=protein.header, sequence=protein.sequence) for protein in proteins]


def digest_proteins(
    proteins: Sequence[ProteinRecord],
    special_amino_acids: Sequence[str] = DEFAULT_SPECIAL_AMINO_ACIDS,
    length_range: tuple[int, int] = DEFAULT_LENGTH_RANGE,
) -> list[str]:
    processor = PeptideProcessor(list(special_amino_acids))
    min_length, max_length = length_range
    peptides: dict[str, None] = {}

    for protein in proteins:
        for peptide_range in processor.get_all_peptides(protein.sequence):
            peptide = protein.sequence[peptide_range.start:peptide_range.stop]
            if min_length <= len(peptide) <= max_length and has_only_canonical_amino_acids(peptide):
                peptides.setdefault(peptide, None)

    return list(peptides)


def get_peptide_sequences_from_target_fasta(
    input_fasta: str,
    decoy_generator_method: str,
    seed: int = 42,
    length_range: tuple[int, int] = DEFAULT_LENGTH_RANGE,
) -> tuple[list[str], list[str]]:
    target_proteins = read_protein_records(input_fasta)
    decoy_proteins = _protein_records_from_fasta_records(
        _create_generator(decoy_generator_method, seed=seed).convert_fasta(
            iter(_fasta_records_from_protein_records(target_proteins))
        )
    )
    return (
        digest_proteins(target_proteins, length_range=length_range),
        digest_proteins(decoy_proteins, length_range=length_range),
    )


def get_peptide_sequences_from_labeled_fasta(
    input_fasta: str,
    length_range: tuple[int, int] = DEFAULT_LENGTH_RANGE,
) -> tuple[list[str], list[str]]:
    target_proteins, decoy_proteins = split_target_and_decoy_proteins(input_fasta)
    return (
        digest_proteins(target_proteins, length_range=length_range),
        digest_proteins(decoy_proteins, length_range=length_range),
    )


def get_target_peptide_sequences_from_fasta(
    input_fasta: str,
    length_range: tuple[int, int] = DEFAULT_LENGTH_RANGE,
) -> list[str]:
    return digest_proteins(read_protein_records(input_fasta), length_range=length_range)


def _payload(
    name: str,
    target_peptides: list[str],
    decoy_peptides: list[str] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "target_peptides": target_peptides,
        "decoy_peptides": decoy_peptides,
    }


def build_method_payloads(
    target_fasta: str,
    decoy_methods: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    target_peptides = get_target_peptide_sequences_from_fasta(target_fasta)
    payloads = [_payload("target", target_peptides)]
    for pair_name, decoy_method in decoy_methods:
        _, decoy_peptides = get_peptide_sequences_from_target_fasta(
            input_fasta=target_fasta,
            decoy_generator_method=decoy_method,
        )
        payloads.append(_payload(pair_name, target_peptides, decoy_peptides))
    return payloads


def build_labeled_fasta_payloads(
    labeled_fastas: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for pair_name, fasta_path in labeled_fastas:
        target_peptides, decoy_peptides = get_peptide_sequences_from_labeled_fasta(fasta_path)
        payloads.append(_payload(pair_name, target_peptides, decoy_peptides))
    return payloads


def build_cross_target_payloads(
    reference_fasta: str,
    comparison_fastas: Sequence[tuple[str, str]],
    decoy_methods: Sequence[tuple[str, str]] = (),
) -> list[dict[str, object]]:
    reference_peptides = get_target_peptide_sequences_from_fasta(reference_fasta)
    payloads = [_payload("target", reference_peptides)]
    for pair_name, decoy_method in decoy_methods:
        _, decoy_peptides = get_peptide_sequences_from_target_fasta(
            input_fasta=reference_fasta,
            decoy_generator_method=decoy_method,
        )
        payloads.append(_payload(pair_name, reference_peptides, decoy_peptides))
    for pair_name, fasta_path in comparison_fastas:
        payloads.append(_payload(pair_name, reference_peptides, get_target_peptide_sequences_from_fasta(fasta_path)))
    return payloads


def plot_auc_bars(
    stats: pd.DataFrame,
    out_pdf: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    label_map: dict[str, str] | None = None,
) -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required to plot AUC bars")
    labels = [
        (label_map.get(name, name) if label_map else name).replace("_", " ")
        for name in stats["experiment"]
    ]
    positions = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(max(8, len(stats) * 1.2), 5))
    ax.bar(positions, stats["auc_mean"], yerr=stats["auc_std"], capsize=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Validation AUC")
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.show()
    plt.close(fig)


def format_auc_value(value: float) -> str:
    return f"{math.floor(float(value) * 100.0) / 100.0:.2f}"


def format_class_count(value: float) -> str:
    return f"{float(value) / 1000.0:.1f}"


def plot_auc_heatmap(stats, out_pdf, rows, columns, title, cmap="viridis", vmin=0.45, vmax=0.85):
    values = load_heatmap_values(stats)
    auc_matrix, class_a_matrix = build_heatmap_matrices(values, rows=rows, columns=columns)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(auc_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(list(columns), color="black")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(rows), color="black")
    ax.set_xlabel("Decoy generation mode", color="black")
    ax.set_ylabel("ESM model size", color="black")
    ax.set_title(title, color="black")
    ax.tick_params(colors="black")
    for row_idx in range(auc_matrix.shape[0]):
        for col_idx in range(auc_matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{format_auc_value(auc_matrix[row_idx, col_idx])}\n{format_class_count(class_a_matrix[row_idx, col_idx])}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                linespacing=1.2,
            )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("AUC mean", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.get_yticklabels(), color="black")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.show()
    plt.close(fig)


def load_heatmap_values(stats: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    required = {"experiment", "auc_mean", "class_a"}
    missing_columns = sorted(required - set(stats.columns))
    if missing_columns:
        raise ValueError(f"Stats DataFrame must contain columns {missing_columns}")

    values: dict[tuple[str, str], tuple[float, float]] = {}
    for row in stats.to_dict(orient="records"):
        match = EXPERIMENT_RE.match(str(row["experiment"]).strip())
        if not match:
            continue
        key = (match.group("row"), VARIANT_TO_COLUMN.get(match.group("variant"), ""))
        if not key[1]:
            continue
        if key in values:
            raise ValueError(f"Duplicate heatmap entry for row={key[0]!r}, column={key[1]!r}")
        values[key] = (float(row["auc_mean"]), float(row["class_a"]))
    return values


def build_heatmap_matrices(
    values: dict[tuple[str, str], tuple[float, float]],
    rows: Sequence[str],
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    auc_matrix = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    class_a_matrix = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    missing: list[str] = []

    for row_idx, row_name in enumerate(rows):
        for col_idx, column_name in enumerate(columns):
            key = (row_name, column_name)
            if key not in values:
                missing.append(f"{row_name}.{column_name}")
                continue
            auc_value, class_a_value = values[key]
            auc_matrix[row_idx, col_idx] = auc_value
            class_a_matrix[row_idx, col_idx] = class_a_value

    if missing:
        raise ValueError(f"Missing heatmap values for: {', '.join(missing)}")

    return auc_matrix, class_a_matrix
