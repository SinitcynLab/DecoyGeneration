import math
import os
import random

import torch

torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from typing import NamedTuple, Iterator, List, Tuple
from transformers import EsmTokenizer, EsmForMaskedLM


# https://www.uniprot.org/help/fasta-headers
class FastaRecord(NamedTuple):
    head: str
    sequence: str


def read_fasta_file(filename: str) -> Iterator[FastaRecord]:
    head: str = ""
    sequence: List[str] = []
    with open(filename) as filestream:
        for line in filestream:
            if line.startswith('>'):
                if head != "":
                    yield FastaRecord(
                        head=head,
                        sequence="".join(sequence)
                    )
                head = line[1:-1]
                sequence.clear()
            else:
                sequence.append(line.rstrip())

    if head != "":
        yield FastaRecord(
            head=head,
            sequence="".join(sequence)
        )


def write_fasta_file(filename: str, fasta_records: Iterator[FastaRecord], width: int = 60):
    with open(filename, 'w') as filestream:
        for record in fasta_records:
            filestream.write(f">{record.head}\n")
            for idx in range(0, len(record.sequence), width):
                filestream.write(f"{record.sequence[idx: idx + width]}\n")


def batch_fasta_stream(fasta_records: Iterator[FastaRecord], batch_size: int) -> Iterator[List[FastaRecord]]:
    current_batch: List[FastaRecord] = []
    for fasta_record in fasta_records:
        current_batch.append(fasta_record)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


class EsmModel:
    model: EsmForMaskedLM
    tokenize: EsmTokenizer

    def __init__(self, local_path: str):
        self.model = EsmForMaskedLM.from_pretrained(local_path, local_files_only=True)
        self.tokenizer = EsmTokenizer.from_pretrained(local_path, local_files_only=True)
        self.model.eval()


def get_masking_positions(sequence: str, special_aas: List[str], mask_percent: float) -> Iterator[int]:
    if len(sequence) == 0:
        return
    special_aas_positions: List[int] = []
    if sequence[0] == "M":  # special case of the first amino acid in proteins, which is usually M
        special_aas_positions.append(0)
    else:
        special_aas_positions.append(-1)

    for idx, aa in enumerate(sequence):
        if aa in special_aas:
            special_aas_positions.append(idx)
    special_aas_positions.append(len(sequence))

    for idx in range(0, len(special_aas_positions) - 1):
        start: int = special_aas_positions[idx] + 1
        end: int = special_aas_positions[idx + 1]
        n: int = end - start
        m: int = math.ceil(n * mask_percent)
        if m == 0:
            continue
        for i in sorted(random.sample(range(start, end), m)):
            yield i


def batch_process(
        fasta_records_batch: List[FastaRecord],
        model: EsmModel,
        special_aas: List[str],  # amino acids recognized by protease
        mask_percent: float,  # should be between 0.0 and 1.0
) -> Iterator[FastaRecord]:
    # Canonical 20 amino acids
    canonical_aas = list("ACDEFGHIKLMNPQRSTVWY")
    aa_ids = model.tokenizer.convert_tokens_to_ids(canonical_aas)

    topk: int = 2 + len(special_aas)  # why 2 - aa itself and I/L dillema

    sequence_batch: List[str] = [record.sequence for record in fasta_records_batch]
    inputs = model.tokenizer(sequence_batch, return_tensors="pt", padding=True)  # [batch_size, L, vocab]
    mask_positions: List[List[int]] = [[] for _ in range(len(fasta_records_batch))]
    for sequence_idx, sequence in enumerate(sequence_batch):
        for mask_idx in get_masking_positions(sequence, special_aas, mask_percent):
            inputs["input_ids"][sequence_idx][mask_idx] = model.tokenizer.mask_token_id
            mask_positions[sequence_idx].append(mask_idx)

    with torch.no_grad():
        outputs = model.model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)  # [batch_size, L, vocab]

    for sequence_idx, sequence in enumerate(sequence_batch):
        new_sequence: List[str] = list(sequence)
        for mask_position in mask_positions[sequence_idx]:
            _, top_idx = torch.topk(probs[sequence_idx, mask_position, aa_ids], k=topk)
            original_aa: str = sequence[mask_position]
            for idx in top_idx:
                new_aa: str = canonical_aas[idx]
                if new_aa == original_aa:
                    continue
                if new_aa in special_aas:
                    continue
                if (new_aa == 'I' and original_aa == 'L') or (
                        new_aa == 'L' and original_aa == 'I'):
                    continue
                new_sequence[mask_position] = new_aa
                break
        record: FastaRecord = fasta_records_batch[sequence_idx]
        yield FastaRecord(
            head=record.head,
            sequence="".join(new_sequence)
        )


def process(
        fasta_records: Iterator[FastaRecord],
        model: EsmModel,
        special_aas: List[str],  # amino acids recognized by protease
        mask_percent: float,  # should be between 0.0 and 1.0
        sort_optimization: bool = True,
        batch_size: int = 256
) -> Iterator[FastaRecord]:
    # Sort by length to simplify the padding task
    if sort_optimization:
        fasta_records_list: List[FastaRecord] = list(fasta_records)
        fasta_records_out_list: List[Tuple[int, FastaRecord]] = []
        sort_idx: List[int] = sorted(range(0, len(fasta_records_list)),
                                     key=lambda idx: len(fasta_records_list[idx].sequence))
        for i in range(0, len(fasta_records_list), batch_size):
            tmp: List[FastaRecord] = []
            for record in batch_process(
                    [fasta_records_list[idx] for idx in sort_idx[i: i + batch_size]],
                    model,
                    special_aas,
                    mask_percent
            ):
                tmp.append(record)
            for idx, record in zip(sort_idx[i: i + batch_size], tmp):
                fasta_records_out_list.append((idx, record))
        fasta_records_out_list.sort()
        for idx, record in fasta_records_out_list:
            yield record

    else:
        for fasta_records_batch in batch_fasta_stream(fasta_records, batch_size):
            yield from batch_process(
                fasta_records_batch,
                model,
                special_aas,
                mask_percent
            )


def main(fasta_filename: str, fasta_out_filename: str, esm_folder: str):
    assert os.path.isfile(fasta_filename)
    assert os.path.isdir(esm_folder)  # if False, load it with huggingface
    random.seed(42)
    write_fasta_file(
        fasta_out_filename,
        process(
            fasta_records=read_fasta_file(fasta_filename),
            model=EsmModel(esm_folder),
            special_aas=list("RK"),
            mask_percent=0.3,
            sort_optimization=True,
            batch_size=64
        )
    )


if __name__ == "__main__":
    main("data/UP000000625_83333.fasta", "data/UP000000625_83333.out.fasta", "models/esm2_t6_8M_UR50D")
