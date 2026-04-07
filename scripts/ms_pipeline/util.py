from pathlib import Path
import hashlib
import os

from typing import List


def plot_ext() -> str:
    """Return '.pdf' if PLOT_PDFS=1, else '.png'."""
    return ".pdf" if os.environ.get("PLOT_PDFS") == "1" else ".png"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def split_proteins_field(s: str) -> List[str]:
    # Sage uses semicolon-separated proteins in TSV (commonly). We accept common separators.
    if s is None:
        return []
    s = str(s)
    if s.strip() == "":
        return []
    # prefer ';', then ',', then whitespace
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
    elif "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    # take only accession token (up to whitespace) for each part
    return [p.split()[0] for p in parts]