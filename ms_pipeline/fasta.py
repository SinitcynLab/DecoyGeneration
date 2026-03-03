from dataclasses import dataclass

from typing import Iterable, List, Optional
from pathlib import Path

import re


@dataclass
class FastaRecord:
    header: str
    sequence: str

def read_fasta(path: Path) -> Iterable[FastaRecord]:
    header: Optional[str] = None
    seq_chunks: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header=header, sequence="".join(seq_chunks).upper())
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(re.sub(r"\s+", "", line))
        if header is not None:
            yield FastaRecord(header=header, sequence="".join(seq_chunks).upper())


def write_fasta(records: Iterable[FastaRecord], path: Path, wrap: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(f">{rec.header}\n")
            seq = rec.sequence
            for i in range(0, len(seq), wrap):
                fh.write(seq[i : i + wrap] + "\n")


def parse_accession(header: str) -> str:
    """
    Extract a protein accession from a FASTA header.
    Heuristic: take up to first whitespace.
    """
    return header.split()[0]
