"""Dataset utilities for peptide sequences.

Supports two input formats:
1) Plain peptide per line: "PEPTIDE"
2) Annotated lines produced by prepare_data:
        "<OS=[Species]> <LEN=N> <sp>PEPTIDE<ep>"

Parsing is tolerant to legacy wrappers and will also accept "<s>SEQ</s>".

No auxiliary feature bits are produced. The dataset returns only token ids,
optionally with metadata.

Metadata retention:
- By default, items return only a tensor of token ids suitable for training.
- Set ``return_meta=True`` in :class:`PeptideDataset` to also get a metadata dict
    per item with keys: {"species", "sequence"}.
    The collator will include a 'meta' list in the batched output when present.
"""
from __future__ import annotations
from typing import List, Sequence, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from .vocab import Vocab, DEFAULT_VOCAB
import re


def parse_annotated_line(line: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Parse annotated line into (species, length, sequence).

    Accepts flexible formats; extracts sequence if one of the wrappers is present:
    - Preferred: '<sp>SEQ<ep>' (from prepare_data.py)
    - Legacy: '<s>SEQ</s>'

    Species and length are optional: '<OS=[..]>' and '<LEN=N>' if present.
    Returns (species, length, sequence) or (None, None, None) when no wrapper found.
    """
    line = line.strip()
    # Prefer prepare_data wrapper first, then legacy
    m_len = re.search(r"<LEN=(\d+)>", line)
    length = int(m_len.group(1)) if m_len else None
    # Remove <LEN=...> from line, rest is sequence
    seq = re.sub(r"<LEN=\d+>", "", line).strip()
    seq = re.sub(" ", "", seq)
    return length, seq

class PeptideDataset(Dataset):
    def __init__(self, lines: Sequence[str], vocab: Vocab = DEFAULT_VOCAB, max_len: int = 64, *, return_meta: bool = False, prepend_species_token: bool = False):
        self.lines = lines
        self.vocab = vocab
        self.max_len = max_len
        self.return_meta = return_meta
        self.prepend_species_token = prepend_species_token
        # Build entries
        self.entries: List[Tuple[str, Optional[str], Optional[int]]] = []  # (seq, species, length)
        for line in lines:
            ln, seq = parse_annotated_line(line)
            if seq is None:
                seq = line.strip()
                ln = None
            self.entries.append((seq, ln))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        seq, length = self.entries[idx]
        # Do not strip tags: caller's vocabulary contains special tokens.
        # We rely on parse_annotated_line to have extracted the peptide sequence already.
        # Optionally prepend species token for conditional training/generation
        seq_for_encoding = seq
        ids = self.vocab.encode(seq_for_encoding, add_bos=True, add_eos=True)
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            ids[-1] = self.vocab.eos_id
        if self.return_meta:
            meta: Dict[str, Any] = {
                'length': length,
                'sequence': seq,
            }
            return torch.tensor(ids, dtype=torch.long), meta
        else:
            return torch.tensor(ids, dtype=torch.long)

def collate_batch(batch, pad_id: int):
    """Collate a batch of items into padded tensors.

    Accepts items of shape:
    - ids
    - (ids, meta) when dataset was created with return_meta=True
    In the latter case, a 'meta' list is included in the returned dict.
    """
    # Normalize batch to (ids, optional_meta)
    ids_list: List[torch.Tensor] = []
    meta_list: Optional[List[Dict[str, Any]]] = None
    first = batch[0]
    if isinstance(first, torch.Tensor):
        ids_list = list(batch)
    else:
        # (ids, meta)
        meta_list = []
        for item in batch:
            ids_list.append(item[0])
            meta_list.append(item[1])
    max_len = max(x.size(0) for x in ids_list)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(ids_list):
        L = seq.size(0)
        input_ids[i, :L] = seq
        attention_mask[i, :L] = 1
    # targets are next token prediction: shift
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_id
    batch_out = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    if meta_list is not None:
        batch_out['meta'] = meta_list
    return batch_out
