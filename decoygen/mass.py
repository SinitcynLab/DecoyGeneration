"""Monoisotopic mass computation and composition utilities."""
from __future__ import annotations
from typing import Dict

AA_MONO_MASS: Dict[str, float] = {
    'A': 71.03711,
    'C': 103.00919,
    'D': 115.02694,
    'E': 129.04259,
    'F': 147.06841,
    'G': 57.02146,
    'H': 137.05891,
    'I': 113.08406,
    'K': 128.09496,
    'L': 113.08406,
    'M': 131.04049,
    'N': 114.04293,
    'P': 97.05276,
    'Q': 128.05858,
    'R': 156.10111,
    'S': 87.03203,
    'T': 101.04768,
    'V': 99.06841,
    'W': 186.07931,
    'Y': 163.06333,
}
# Add water mass for full peptide (H2O)
WATER_MASS = 18.01056

def monoisotopic_mass(peptide: str) -> float:
    mass = WATER_MASS
    for aa in peptide:
        mass += AA_MONO_MASS.get(aa, 0.0)
    return mass

def composition(peptide: str) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in AA_MONO_MASS}
    for aa in peptide:
        if aa in counts:
            counts[aa] += 1
    return counts

def identity_fraction(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    # simple global identity fraction by aligned positions (truncate to min length)
    L = min(len(seq1), len(seq2))
    matches = sum(1 for a, b in zip(seq1[:L], seq2[:L]) if a == b)
    return matches / L if L > 0 else 0.0
