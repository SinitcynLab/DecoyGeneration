"""Decoding constraints for tryptic rules and dynamic masking.

Implements:
 - Length constraints (min / max)
 - Tryptic end enforcement: peptide should end with K/R unless length forced
 - Missed cleavage counting: internal (non-terminal) K/R followed by !P counts towards missed unless cleavage taken
 - Prevention of more than allowed missed cleavages by masking further non-cleavage additions after threshold
"""
from __future__ import annotations
from typing import List

from .vocab import DEFAULT_VOCAB

TRYPTIC_CLEAVAGE_AFTER = {"K", "R"}

class ConstraintState:
    def __init__(self, max_missed: int):
        self.max_missed = max_missed
        self.missed = 0
        self.sequence: List[str] = []

    def update(self, seq: List[str]):
        self.sequence = seq
        self.missed = count_missed_cleavages(seq)

    def can_add(self, next_aa: str) -> bool:
        # If already exceeded missed cleavages, restrict adding more residues unless next is a cleavage terminator (K/R) that could finish later.
        if self.missed >= self.max_missed:
            # Allow adding cleavage residue to potentially terminate; block others.
            return next_aa in TRYPTIC_CLEAVAGE_AFTER
        return True

def count_missed_cleavages(seq: List[str]) -> int:
    # Internal K/R not followed by P and not at end counts as missed cleavage.
    missed = 0
    for i, aa in enumerate(seq[:-1]):
        if aa in TRYPTIC_CLEAVAGE_AFTER:
            nxt = seq[i+1]
            if nxt != 'P':
                # cleavage site exists; if peptide continues, it's missed (unless this is last index which it isn't here)
                missed += 1
    # Terminal site shouldn't count; subtract if last aa is K/R (and not followed by P obviously)
    if seq and seq[-1] in TRYPTIC_CLEAVAGE_AFTER:
        # it was counted in loop only if len>1 and previous char K/R; adjust logic: easier to recompute precisely
        # Recompute with terminal exclusion cleanly
        missed = 0
        for i, aa in enumerate(seq[:-1]):
            if aa in TRYPTIC_CLEAVAGE_AFTER and seq[i+1] != 'P':
                # If cleavage not taken (sequence continues past this site), count
                missed += 1
    return missed

def apply_length_constraints(step: int, min_length: int, max_length: int, vocab_size: int, logits, eos_id: int):
    if step < min_length:
        logits[eos_id] = -1e9
    if step >= max_length - 1:  # force termination
        for i in range(vocab_size):
            if i != eos_id:
                logits[i] = -1e9

def enforce_tryptic_end(seq: List[str], step: int, min_length: int, max_length: int, logits, eos_id: int, vocab):
    # If we are allowed to end (>= min_length) but current last AA is not K/R, down-weight EOS (not hard mask to allow flexibility near max length)
    if step >= min_length and step < max_length - 1:
        if not seq or seq[-1] not in TRYPTIC_CLEAVAGE_AFTER:
            logits[eos_id] -= 5.0  # discourage early end without K/R
    # If at final step (max_length-1) force EOS regardless
    if step >= max_length - 1:
        return

def mask_invalid_tokens(seq_ids, logits, vocab, tryptic: bool, state: ConstraintState):
    # Convert ids (excluding BOS) to sequence tokens
    seq_tokens = [vocab.id_to_token[i] for i in seq_ids if vocab.id_to_token[i] not in ("<s>", "<pad>")]
    state.update([t for t in seq_tokens if t not in ("</s>")])
    if tryptic and state.missed >= state.max_missed:
        # Mask adding non-cleavage residues further (except EOS and K/R)
        for idx, tok in vocab.id_to_token.items():
            if tok in ("<pad>", "<s>"):
                logits[idx] = -1e9
            elif tok not in ("</s>") and tok not in TRYPTIC_CLEAVAGE_AFTER:
                logits[idx] -= 5.0  # discourage but not fully forbid
    return logits
