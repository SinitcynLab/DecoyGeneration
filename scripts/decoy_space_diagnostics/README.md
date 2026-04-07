Script to run spectral-level decoy diagnostics.

To run:

```bash
python3 diagnostics.py \
  --fasta /path/to/library.fasta \
  --outdir /path/to/output \
  --workers 16 \
  --parallel-batch-size 512 \
  --koina-url https://koina.wilhelmlab.org \
  --koina-model Prosit_2020_intensity_HCD \
  --precursor-charge 2 \
  --collision-energy 27 \
  --koina-batch-size 1000 \
  --top-suspicious 100000 \
  --random-null-queries 1000000
```
