from .fasta import read_fasta

from typing import List, Tuple, Optional
from pathlib import Path


def trypsin_cleave_sites(seq: str, cleave_at: str = "KR", restrict: str = "P", c_terminal: bool = True) -> List[int]:
    """
    Return cleavage positions as indices in [0..len(seq)].
    For trypsin-like digestion: cleave after K/R not followed by P (C-terminal cleavage).
    """
    n = len(seq)
    cuts = [0]
    if c_terminal:
        for i, aa in enumerate(seq[:-1]):  # last cannot be a cleavage site with following aa check
            if aa in cleave_at and seq[i + 1] != restrict:
                cuts.append(i + 1)
        cuts.append(n)
    else:
        # N-terminal cleavage before matching AA (rarely used); restrict means do not cleave if preceding aa is restrict.
        for i, aa in enumerate(seq[1:], start=1):
            if aa in cleave_at and seq[i - 1] != restrict:
                cuts.append(i)
        cuts.append(n)
    # ensure sorted unique
    cuts = sorted(set(cuts))
    return cuts


def digest_sequence(
    seq: str,
    *,
    enzyme_cleave_at: str,
    restrict: str,
    missed_cleavages: int,
    min_len: int,
    max_len: int,
    c_terminal: bool = True,
    cleave_at_special: Optional[str] = None,
) -> List[str]:
    """
    Produce peptide sequences given digestion parameters.

    Special behavior matching Sage docs:
      - cleave_at == "$": no digestion, use FASTA entries as-is
      - cleave_at == "": non-enzymatic, generate all substrings between min_len..max_len (very expensive)
    """
    if cleave_at_special == "$":
        pep = seq
        if min_len <= len(pep) <= max_len:
            return [pep]
        return []
    if enzyme_cleave_at == "":
        # Non-enzymatic: all substrings in length range
        peps: List[str] = []
        n = len(seq)
        for i in range(n):
            for L in range(min_len, max_len + 1):
                j = i + L
                if j <= n:
                    peps.append(seq[i:j])
        return peps

    cuts = trypsin_cleave_sites(seq, cleave_at=enzyme_cleave_at, restrict=restrict, c_terminal=c_terminal)
    peps: List[str] = []
    # peptides are between cuts[k]..cuts[m], where m <= k+missed_cleavages+1
    for k in range(len(cuts) - 1):
        for m in range(k + 1, min(len(cuts), k + missed_cleavages + 2)):
            pep = seq[cuts[k] : cuts[m]]
            if min_len <= len(pep) <= max_len:
                peps.append(pep)
    return peps


def digest_fasta_unique_peptides(
    fasta_path: Path,
    *,
    enzyme_cleave_at: str = "KR",
    restrict: str = "P",
    missed_cleavages: int = 2,
    min_len: int = 7,
    max_len: int = 50,
    c_terminal: bool = True,
    collapse_il: bool = False,
    cleave_at_special: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Return (#unique peptides, #total peptides) for a FASTA with the given digestion settings.

    Used to estimate r = |E|/|T| as an "effective" database-size ratio, on peptide space.
    """
    uniq = set()
    total = 0
    for rec in read_fasta(fasta_path):
        seq = rec.sequence
        if collapse_il:
            seq = seq.replace("I", "L")
        peps = digest_sequence(
            seq,
            enzyme_cleave_at=enzyme_cleave_at,
            restrict=restrict,
            missed_cleavages=missed_cleavages,
            min_len=min_len,
            max_len=max_len,
            c_terminal=c_terminal,
            cleave_at_special=cleave_at_special,
        )
        for p in peps:
            total += 1
            uniq.add(p)
    return len(uniq), total
