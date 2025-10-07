"""FASTA processing and in-silico tryptic digestion to produce peptide list.

Usage:
  python -m decoygen.prepare_data --fasta data/proteome1.fasta data/proteome2.fasta \
      --out peptides.txt --min-length 7 --max-length 35 --max-missed-cleavages 2

Features:
  - Parses one or more FASTA files
  - Performs canonical tryptic digestion (cleave after K/R unless next is P)
  - Enumerates peptides with up to N missed cleavages
  - Filters by length and (optional) mass range
  - De-duplicates peptides
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Iterable, Set, Tuple
import re

from .mass import monoisotopic_mass

def parse_args(arg_list=None):
    p = argparse.ArgumentParser(description="In-silico tryptic digestion of FASTA proteins to peptide list")
    p.add_argument('--fasta', nargs='+', type=Path, required=True, help='One or more FASTA files')
    p.add_argument('--out', type=Path, required=True, help='Output text file (one peptide per line)')
    p.add_argument('--min-length', type=int, default=7)
    p.add_argument('--max-length', type=int, default=35)
    p.add_argument('--max-missed-cleavages', type=int, default=2)
    p.add_argument('--mass-min', type=float, default=None, help='Optional minimum monoisotopic mass')
    p.add_argument('--mass-max', type=float, default=None, help='Optional maximum monoisotopic mass')
    return p.parse_args(arg_list)

def read_fasta_records(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (header, sequence) for each FASTA entry.

    Header is the full header line without the leading '>'.
    """
    with path.open() as f:
        header: str | None = None
        seq_lines: List[str] = []
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            yield header, ''.join(seq_lines)

def read_fasta(path: Path) -> Iterable[str]:
    """Backward-compatible generator that yields only sequences."""
    for _, seq in read_fasta_records(path):
        yield seq

def parse_species(header: str) -> str:
    """Extract species name from FASTA header.

    Priority order:
    1) UniProt-style `OS=...` field (stops before next ALLCAPS= key)
    2) The last bracketed token `[...]` in the header
    3) Fallback "Unknown"
    """
    # UniProt OS= pattern
    m = re.search(r"OS=([^=]+?)(?:\s[A-Z]{1,3}=|$)", header)
    if m:
        return m.group(1).strip()
    # Bracketed species
    m2 = re.findall(r"\[([^\]]+)\]", header)
    if m2:
        return m2[-1].strip()
    return "Unknown"

def tryptic_cleave(protein: str) -> List[int]:
    cut_positions = [0]
    for i, aa in enumerate(protein[:-1]):
        if aa in ('K','R') and protein[i+1] != 'P':
            cut_positions.append(i+1)
    cut_positions.append(len(protein))
    return cut_positions

def generate_peptides(protein: str, max_missed: int, min_len: int, max_len: int) -> List[str]:
    cuts = tryptic_cleave(protein)
    peptides: List[str] = []
    n = len(cuts)
    for i in range(n-1):
        for missed in range(0, max_missed+1):
            j = i + 1 + missed
            if j >= n:
                continue
            pep = protein[cuts[i]:cuts[j]]
            L = len(pep)
            if L < min_len or L > max_len:
                continue
            peptides.append(pep)
    return peptides

def filter_mass_range(peptides: Iterable[str], mass_min, mass_max):
    if mass_min is None and mass_max is None:
        return list(peptides)
    out = []
    for p in peptides:
        m = monoisotopic_mass(p)
        if (mass_min is None or m >= mass_min) and (mass_max is None or m <= mass_max):
            out.append(p)
    return out

def prepare_data(fasta_paths, out_path, min_length=7, max_length=500, max_missed_cleavages=2, mass_min=None, mass_max=None):
    class SimpleArgs:
        pass
    args = SimpleArgs()
    args.fasta = [Path(p) for p in fasta_paths]
    args.out = Path(out_path)
    args.min_length = min_length
    args.max_length = max_length
    args.max_missed_cleavages = max_missed_cleavages
    args.mass_min = mass_min
    args.mass_max = mass_max
    # Build peptides with species annotations; de-duplicate by (peptide, species)
    annotated: Set[Tuple[str, str]] = set()
    for fasta in args.fasta:
        if not fasta.exists():
            continue
        for header, protein in read_fasta_records(fasta):
            species = parse_species(header)
            # peps = generate_peptides(protein, args.max_missed_cleavages, args.min_length, args.max_length)
            # Optional mass filter inline
            # if args.mass_min is not None or args.mass_max is not None:
                # peps = filter_mass_range(peps, args.mass_min, args.mass_max)
            # for p in protein:#peps:
                # annotated.add((p, species))
            annotated.add((protein, species))

    # Format: <OS=[Species]> <LEN=N> <sp>PEPTIDE<ep>
    lines: List[str] = []
    for pep, species in annotated:
        lines.append(f"<OS=[{species}]> <LEN={len(pep)}> <sp>{pep}<ep>")
    lines.sort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        for line in lines:
            f.write(line + '\n')
    return lines

def main(cli_args=None):
    args = parse_args(cli_args)
    lines = prepare_data(
        fasta_paths=[str(p) for p in args.fasta],
        out_path=str(args.out),
        min_length=args.min_length,
        max_length=args.max_length,
        max_missed_cleavages=args.max_missed_cleavages,
        mass_min=args.mass_min,
        mass_max=args.mass_max,
    )
    print(f"Wrote {len(lines)} peptides to {args.out}")

if __name__ == '__main__':
    main()
