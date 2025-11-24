from typing import List

def split_targets(target_sequences: List[str]):
    N = len(target_sequences)//2
    targets = target_sequences[0:N]
    pretend_decoys = target_sequences[N:N*2]

    return targets, pretend_decoys

def remove_long_sequences(sequences: List[str], cap_length: int = 10_000) -> List[str]:
    for i, _ in enumerate(sequences):
        if len(sequences[i]) > cap_length:
            sequences.pop(i)
    return sequences