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


def write_fasta_file(filename: str, fasta_records: Iterator[FastaRecord], width: int = 60, write_mode: str = 'w', label: str = ""):
    with open(filename, write_mode) as filestream:
        for record in fasta_records:
            record.head = label + record.head # label sequence
            filestream.write(f">{record.head}\n")
            for idx in range(0, len(record.sequence), width):
                filestream.write(f"{record.sequence[idx: idx + width]}\n")