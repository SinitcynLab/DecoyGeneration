from typing import List

def remove_long_sequences(sequences: List[str], cap_length: int = 10_000) -> List[str]:
    for i, _ in enumerate(sequences):
        if len(sequences[i]) > cap_length:
            sequences.pop(i)
    return sequences