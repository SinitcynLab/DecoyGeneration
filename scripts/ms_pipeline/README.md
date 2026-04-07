Set of scripts to run various DDA MS/MS pipelines and perform statistical analyses on the results.

Example runs:

```bash
python3 -m ms_pipeline.ms_pipeline \
  --spectra /path/to/spectra.mzML \
  --target-fasta /path/to/library.fasta \
  --entrapment-mode none \
  --experiment-name some_name \
  --output-dir /path/to/output/dir \
  --sage-bin /path/to/sage

python3 -m ms_pipeline.ms_pipeline \
  --spectra /path/to/spectra.mzML \
  --target-fasta /path/to/library.fasta \
  --entrapment-mode foreign \
  --entrapment-prefix ENTRAP_ \
  --entrapment-fasta /path/to/entrapment/library.fasta \
  --experiment-name some_name \
  --output-dir /path/to/output/dir \
  --sage-bin /path/to/sage
```
