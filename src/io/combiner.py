
from typing import Iterable, List
from src.io.fasta import FastaRecord, read_fasta_file, write_fasta_file

class Combiner(object):
    def __init__(self):
        object.__init__(self)

    def combine(in_file_names: Iterable[str], out_file_name: str):
        all_records: List[FastaRecord] = []
        for file_name in in_file_names:
            file_records = [record for record in read_fasta_file(file_name)]
            all_records += file_records
        write_fasta_file(out_file_name, all_records)