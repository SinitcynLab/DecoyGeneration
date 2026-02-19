from typing import Iterator, List, NamedTuple


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


def write_fasta_file(filename: str, fasta_records: Iterator[FastaRecord], width: int = 60, prefix: str = ""):
    with open(filename, "w") as filestream:
        write_fasta(filestream, fasta_records, width=width, prefix=prefix)


def write_fasta(filestream, fasta_records: Iterator[FastaRecord], width: int = 60, prefix: str = ""):
    for record in fasta_records:
        if prefix != "":
            filestream.write(f">{prefix}{record.head}\n")
        else:
            filestream.write(f">{record.head}\n")
        for idx in range(0, len(record.sequence), width):
            filestream.write(f"{record.sequence[idx: idx + width]}\n")
