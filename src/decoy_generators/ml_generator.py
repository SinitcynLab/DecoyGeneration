import math

from typing import Iterator, List, Tuple
from random import Random
from enum import Enum

from src.decoy_generators.decoy_generator import DecoyGenerator

class MlGeneratorType(Enum):
    BEST = 1,
    WORST = 2

class MlGenerator(DecoyGenerator):
    def __init__(
            self,
            local_path: str,
            random: Random,
            special_amino_acids: List[str],
            mask_percent: float = 0.3,  # should be between 0.0 and 1.0
            sort_optimization: bool = True,
            batch_size: int = 64,
            ml_generator_type: MlGeneratorType = MlGeneratorType.BEST
    ):
        DecoyGenerator.__init__(self, special_amino_acids)
        self.random = random
        self.mask_percent = mask_percent
        self.sort_optimization = sort_optimization
        self.batch_size = batch_size
        self.esm_generator_type = ml_generator_type
        self.local_path = local_path

    def _batch_convert(self, target_batch: List[str]) -> Iterator[str]:
        raise NotImplementedError()

    def _get_masked_positions(self, sequence: str):
        positions: List[int] = list(self.get_positions_special_aas(sequence))
        for idx in range(1, len(positions)):
            start: int = positions[idx - 1] + 1
            end: int = positions[idx]
            n: int = end - start
            m: int = math.ceil(n * self.mask_percent)
            if m == 0:
                continue
            for i in sorted(self.random.sample(range(start, end), m)):
                yield i

    @staticmethod
    def _batch(a: Iterator[str], batch_size: int) -> Iterator[List[str]]:
        b: List[str] = []
        n: int = 0
        for item in iter(a):
            b.append(item)
            n += 1
            if n % batch_size == 0:
                yield b
                b = []
        yield b

    def convert(self, targets: Iterator[str]) -> Iterator[str]:
        if self.sort_optimization:
            target_list: List[str] = list(targets)
            target_out_list: List[Tuple[int, str]] = []
            sort_idx: List[int] = sorted(range(0, len(target_list)), key=lambda idx: len(target_list[idx]))
            for i in range(0, len(target_list), self.batch_size):
                tmp: List[str] = [
                    _ for _ in self._batch_convert([target_list[idx] for idx in sort_idx[i: i + self.batch_size]])
                ]
                for idx, record in zip(sort_idx[i: i + self.batch_size], tmp):
                    target_out_list.append((idx, record))
            target_out_list.sort()
            for idx, record in target_out_list:
                yield record
        else:
            for fasta_records_batch in self._batch(targets, self.batch_size):
                yield from self._batch_convert(fasta_records_batch)