# Decoy Peptide Generator (Transformer)

Decoder-only Transformer language model for generating proteomics decoy peptide sequences under biochemical constraints.

## Features
- Amino acid vocabulary with special tokens
- Decoder-only Transformer with causal masking
- Tryptic / missed cleavage aware constrained decoding
- Nucleus (top-p) sampling with temperature & repetition penalty
- Mass & composition filtering
- Removal of overlapping / identical target peptides

### Encoder-only ESM Test   
The test for the encoder-only ESM model is located in `esm_test.py`.
 
## Quick Start
```bash
pip install -r requirements.txt
python -m decoygen.train --data peptides.txt --out-dir runs/exp1
python -m decoygen.generate --checkpoint runs/exp1/best.pt --num 1000 --targets targets.txt --out decoys.fasta
```

## License
MIT
