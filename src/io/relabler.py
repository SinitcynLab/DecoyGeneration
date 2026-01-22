
from typing import List
from src.io.fasta import FastaRecord, read_fasta_file, write_fasta_file

class Relabler(object):
    def __init__(self):
        object.__init__(self)

    def relabel(label:str, in_file_name: str, out_file_name: str):
        in_records: List[FastaRecord] = read_fasta_file(in_file_name)
        out_records: List[FastaRecord] = []
        for in_record in in_records:
            new_head = label + in_record.head
            out_records.append(FastaRecord(new_head, in_record.sequence))
        write_fasta_file(out_file_name, out_records)