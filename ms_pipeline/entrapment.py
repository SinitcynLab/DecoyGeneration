import pandas as pd

from dataclasses import dataclass

from typing import List, Dict, Tuple, Optional, Iterable
from pathlib import Path
import shutil
import re

from .fasta import FastaRecord, read_fasta, write_fasta, parse_accession
from .protease import digest_sequence
from .util import sha256_file, split_proteins_field


@dataclass
class EntrapmentConfig:
    mode: str  # "none" | "foreign" | "paired_shuffled_peptides"
    decoy_prefix: str = "rev_"
    entrapment_fasta: Optional[Path] = None  # for foreign
    entrapment_prefix: str = "ENTRAP_"
    entrapment_strategy: str = "unambiguous"  # "unambiguous" | "any"
    target_prefix: str = ""  # only used in paired peptide mode
    pairing_regex: str = r"^(TGT|ENTRAP)\|(\d+)$"  # for paired method
    paired_seed: int = 1
    paired_keep_c_terminal: bool = True
    paired_k: int = 1  # k-matched generalization (not fully implemented)
    paired_min_len: int = 7
    paired_max_len: int = 35
    paired_missed_cleavages: int = 2
    paired_digest_cleave_at: str = "KR"
    paired_digest_restrict: str = "P"
    paired_digest_c_terminal: bool = True
    paired_cleave_at_special_for_search: str = "$"  # Sage: no digestion
    collapse_il: bool = False

@dataclass
class EntrapmentLabeling:
    entrapment_prefix: str = "ENTRAP_"
    entrapment_strategy: str = "unambiguous"  # "unambiguous" | "any"
    # For paired peptide DB mode
    pairing_regex: str = r"^(TGT|ENTRAP)\|(\d+)$"



def prefix_entrapment_headers(records: Iterable[FastaRecord], decoy_prefix: str, entrapment_prefix: str) -> Iterable[FastaRecord]:
    for rec in records:
        acc = parse_accession(rec.header)
        new_header = rec.header
        if not acc.startswith(entrapment_prefix):
            # replace only accession token, keep the rest
            parts = rec.header.split(maxsplit=1)
            if parts[0].startswith(decoy_prefix):
                parts[0] = decoy_prefix + entrapment_prefix + parts[0][len(decoy_prefix):]
            else:
                parts[0] = entrapment_prefix + parts[0]
            new_header = parts[0] + ((" " + parts[1]) if len(parts) > 1 else "")
        yield FastaRecord(header=new_header, sequence=rec.sequence)


def build_combined_fasta(
    target_fasta: Path,
    *,
    entrap_cfg: EntrapmentConfig,
    out_fasta: Path,
    work_dir: Path,
) -> Tuple[Path, Dict[str, str]]:
    """
    Build the FASTA to be searched.

    Returns:
      - path to combined fasta
      - metadata dict (for run.json)
    """
    meta: Dict[str, str] = {}
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    if entrap_cfg.mode == "none":
        shutil.copyfile(target_fasta, out_fasta)
        meta["entrapment_mode"] = "none"
        return out_fasta, meta

    if entrap_cfg.mode == "foreign":
        if entrap_cfg.entrapment_fasta is None:
            raise ValueError("entrapment_mode=foreign requires --entrapment-fasta")
        # Write combined: target as-is; entrapment with prefixed accessions
        target_recs = list(read_fasta(target_fasta))
        entrap_recs = list(prefix_entrapment_headers(read_fasta(entrap_cfg.entrapment_fasta), entrap_cfg.decoy_prefix, entrap_cfg.entrapment_prefix))
        write_fasta(target_recs + entrap_recs, out_fasta)
        meta["entrapment_mode"] = "foreign"
        meta["entrapment_prefix"] = entrap_cfg.entrapment_prefix
        meta["target_fasta_sha256"] = sha256_file(target_fasta)
        meta["entrapment_fasta_sha256"] = sha256_file(entrap_cfg.entrapment_fasta)
        return out_fasta, meta

    if entrap_cfg.mode == "paired_shuffled_peptides":
        # Build a *peptide* FASTA: each target peptide becomes one entry; each gets a paired shuffled entrapment peptide.
        # This is intended for Wen et al.'s "paired entrapment" estimator.
        rng = _DeterministicRNG(seed=entrap_cfg.paired_seed)
        # Digest target proteins into peptides
        uniq_peps = []
        seen = set()
        for rec in read_fasta(target_fasta):
            seq = rec.sequence
            if entrap_cfg.collapse_il:
                seq = seq.replace("I", "L")
            peps = digest_sequence(
                seq,
                enzyme_cleave_at=entrap_cfg.paired_digest_cleave_at,
                restrict=entrap_cfg.paired_digest_restrict,
                missed_cleavages=entrap_cfg.paired_missed_cleavages,
                min_len=entrap_cfg.paired_min_len,
                max_len=entrap_cfg.paired_max_len,
                c_terminal=entrap_cfg.paired_digest_c_terminal,
            )
            for p in peps:
                if p not in seen:
                    seen.add(p)
                    uniq_peps.append(p)

        target_records: List[FastaRecord] = []
        entrap_records: List[FastaRecord] = []

        # Avoid collisions: entrapment peptides should not match *any* target peptide,
        # and ideally should be unique among entrapments too. Collisions create shared peptides
        # and break the paired entrapment logic.
        forbidden = set(uniq_peps)

        for i, pep in enumerate(uniq_peps):
            tgt_acc = f"TGT|{i}"
            ent_acc = f"ENTRAP|{i}"
            target_records.append(FastaRecord(header=tgt_acc, sequence=pep))

            ent_pep: Optional[str] = None
            for _attempt in range(50):
                cand = shuffle_peptide_for_entrapment(
                    pep,
                    rng=rng,
                    keep_c_terminal=entrap_cfg.paired_keep_c_terminal,
                )
                if cand not in forbidden:
                    ent_pep = cand
                    break

            if ent_pep is None:
                # Extremely unlikely unless the peptide has many repeated residues.
                # Fall back to reversal; if that collides too, we accept it (and the analysis
                # will report ambiguities in the output tables).
                ent_pep = pep[::-1]

            forbidden.add(ent_pep)
            entrap_records.append(FastaRecord(header=ent_acc, sequence=ent_pep))

        write_fasta(target_records + entrap_records, out_fasta)
        meta["entrapment_mode"] = "paired_shuffled_peptides"
        meta["paired_seed"] = str(entrap_cfg.paired_seed)
        meta["paired_num_target_peptides"] = str(len(target_records))
        meta["paired_cleave_at_special_for_search"] = entrap_cfg.paired_cleave_at_special_for_search
        return out_fasta, meta

    raise ValueError(f"Unknown entrapment mode: {entrap_cfg.mode}")


class _DeterministicRNG:
    """
    A deterministic RNG that avoids global random state.
    Uses Python's random.Random under the hood but avoids importing random at module scope.
    """
    def __init__(self, seed: int):
        import random
        self._r = random.Random(seed)

    def shuffle(self, xs: List[str]) -> None:
        self._r.shuffle(xs)


def shuffle_peptide_for_entrapment(pep: str, *, rng: _DeterministicRNG, keep_c_terminal: bool = True) -> str:
    """
    Create a shuffled entrapment peptide from a target peptide.
    - By default keep the C-terminal residue fixed (tryptic compatibility).
    - Ensure it's different from the original when possible.
    """
    if len(pep) <= 2:
        return pep[::-1]  # trivial fallback

    if keep_c_terminal:
        fixed = pep[-1]
        core = list(pep[:-1])
        # Try a few times to avoid identity
        for _ in range(10):
            rng.shuffle(core)
            cand = "".join(core) + fixed
            if cand != pep:
                return cand
        # if unlucky, return reversed core
        return "".join(core[::-1]) + fixed
    else:
        chars = list(pep)
        for _ in range(10):
            rng.shuffle(chars)
            cand = "".join(chars)
            if cand != pep:
                return cand
        return pep[::-1]

def label_entrapment_rows(
    df: pd.DataFrame,
    *,
    proteins_col: str,
    labeling: EntrapmentLabeling,
) -> pd.DataFrame:
    """
    Add boolean columns:
      - is_entrapment
      - is_target_original
      - is_ambiguous_entrapment
    based on protein accessions in proteins_col.
    """
    ent_prefix = labeling.entrapment_prefix

    def classify(prot_str: str) -> Tuple[bool, bool, bool]:
        prots = split_proteins_field(prot_str)
        if not prots:
            return (False, True, False)  # default to target
        # Remove decoy tag if present (e.g., rev_)
        # We DON'T assume the decoy tag; we just strip common ones for labeling only.
        prots_clean = [re.sub(r"^(rev_|decoy_|DECOY_)", "", p) for p in prots]
        is_ent = [p.startswith(ent_prefix) for p in prots_clean]
        if labeling.entrapment_strategy == "any":
            ent = any(is_ent)
            amb = ent and (not all(is_ent))
            tgt = not ent
            return (ent, tgt, amb)
        # unambiguous: all proteins must be entrapment to count as entrapment
        ent = all(is_ent)
        amb = (any(is_ent) and not all(is_ent))
        tgt = not any(is_ent)  # original target if none are entrapment
        return (ent, tgt, amb)

    out = df.copy()
    prot_series = out[proteins_col].astype(str)
    vals = prot_series.apply(classify)
    out["is_entrapment"] = vals.apply(lambda x: x[0])
    out["is_target_original"] = vals.apply(lambda x: x[1])
    out["is_ambiguous_entrapment"] = vals.apply(lambda x: x[2])
    return out