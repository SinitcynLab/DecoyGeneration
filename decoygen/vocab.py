"""Amino acid vocabulary and tokenization utilities for peptide sequences.

Includes 20 canonical amino acids plus special tokens:
<s> (start), </s> (end), <pad>

Optionally extended to support modified residues (placeholder hook).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable, Optional

SPECIAL_TOKENS = ["<bom>", "<pad>", "<s>", "</s>"] # bom: begining of the message
ORGANISM_TOKENS = ["<OS=[Caenorhabditis elegans]>", "<OS=[Escherichia coli (strain K12)]>", "<OS=[Homo sapiens]>", "<OS=[Saccharomyces cerevisiae (strain ATCC 204508 / S288c)]>"]
AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWYU")  # 20 canonical amino acids 
UNKNOWN_AA = ["X"]  # unknown amino acid token

# Index mapping
ALL_TOKENS =  SPECIAL_TOKENS + ORGANISM_TOKENS + AA_TOKENS + UNKNOWN_AA
BOM, PAD, BOS, EOS = (0, 1, 2, 3)

TOKEN_TO_ID: Dict[str, int] = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID_TO_TOKEN: Dict[int, str] = {i: tok for tok, i in TOKEN_TO_ID.items()}

@dataclass
class Vocab:
    tokens: Optional[List[str]] = None

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = list(ALL_TOKENS)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<s>"]
        self.eos_id = self.token_to_id["</s>"]
        self.bom_id = self.token_to_id["<bom>"]
        # Precompute multi-character tokens for greedy matching (e.g., <OS:...>)
        self._multi_tokens = [t for t in self.tokens if len(t) > 1 and t not in ("<pad>", "<s>", "</s>", "<bom>")]
        # Longest-first to avoid partial matches
        self._multi_tokens.sort(key=len, reverse=True)

    def encode(self, seq: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode a string into token ids.

        Supports greedy matching of known multi-character tokens (e.g., '<OS:...>').
        Falls back to per-character encoding for single-letter amino acids.
        Unknown characters map to the 'X' token id.
        """
        s = seq.strip()
        ids: List[int] = []

        i = 0
        # Rebuild ids to handle special-token normalization (<bom> at start, </s> at end)
        ids = []
        # Greedy match: special tokens + known multi-char tokens (longest-first)
        scan_tokens = list(dict.fromkeys(SPECIAL_TOKENS + self._multi_tokens))
        scan_tokens.sort(key=len, reverse=True)

        while i < len(s):
            matched = False
            for tok in scan_tokens:
                tok_no_space = tok.replace(" ", "")
                if s.startswith(tok, i):
                    ids.append(self.token_to_id[tok])
                    i += len(tok)
                    matched = True
                    break
                if tok_no_space != tok and s.startswith(tok_no_space, i):
                    ids.append(self.token_to_id[tok])
                    i += len(tok_no_space)
                    matched = True
                    break
            if matched:
                continue

            ch = s[i]
            if ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
            else:
            # map unknown to 'X'
                ids.append(self.token_to_id.get("X", self.pad_id))
            i += 1

        # Ensure <bom> is the first token
        if not ids or ids[0] != self.bom_id:
            ids.insert(0, self.bom_id)

        # Ensure </s> is the last token
        if not ids or ids[-1] != self.eos_id:
            ids.append(self.eos_id)
        
        return ids

    def decode(self, ids: Iterable[int], remove_special: bool = False) -> str:
        chars = []
        for i in ids:
            tok = self.id_to_token.get(i, "?")
            if remove_special and tok in SPECIAL_TOKENS:
                continue
            chars.append(tok)
        return "".join(chars)

    @property
    def size(self) -> int:
        return len(self.tokens)

DEFAULT_VOCAB = Vocab()
