"""Filtering utilities for generated decoy peptides."""
from __future__ import annotations
from typing import Iterable, Set, List
from .mass import monoisotopic_mass, identity_fraction


def filter_mass(peptides: Iterable[str], mass_min: float, mass_max: float) -> List[str]:
    return [p for p in peptides if mass_min <= monoisotopic_mass(p) <= mass_max]


def remove_exact_matches(peptides: Iterable[str], targets: Set[str]) -> List[str]:
    return [p for p in peptides if p not in targets]


def remove_high_identity(peptides: Iterable[str], targets: Set[str], threshold: float) -> List[str]:
    kept = []
    for p in peptides:
        ok = True
        for t in targets:
            if identity_fraction(p, t) >= threshold:
                ok = False
                break
        if ok:
            kept.append(p)
    return kept
