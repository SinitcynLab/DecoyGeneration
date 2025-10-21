# Introduction

In mass spectrometry (MS)-based proteomics, peptide identification is typically performed by matching experimental spectra against theoretical spectra derived from protein databases. To estimate the false discovery rate (FDR), target-decoy strategies are widely employed: the search space is augmented with *decoy peptides* that are not expected to exist in the sample. By comparing matches to targets versus decoys, one can statistically control for incorrect identifications.  

The quality of FDR estimation, however, depends critically on the realism of the generated decoys. Current decoy generation methods are simplistic: they often create decoys by reversing amino acid sequences, shuffling them, or applying small permutations decoy. While computationally convenient, such strategies produce peptides that do not reflect the true physicochemical or structural properties of real peptides. Consequently, these decoys may be too easily distinguishable from targets, which can bias FDR estimation and reduce confidence in proteomics studies.


```

## License
MIT
