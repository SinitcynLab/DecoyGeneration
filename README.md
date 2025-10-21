# Generative Decoy Peptides for Proteomics

## Overview

In mass spectrometry (MS)-based proteomics, peptide identification relies on comparing experimental spectra with theoretical spectra derived from protein databases.
To estimate the false discovery rate (FDR), researchers typically use *target-decoy* strategies, where synthetic *decoy peptides* serve as negative controls.
However, current decoy generation methods - such as sequence reversal or random shuffling - are often too simplistic.
They produce unrealistic peptides that may bias FDR estimation and reduce confidence in peptide identification.

This project explores generative AI approaches to create more realistic decoy peptides that better mimic the properties of true peptides while remaining biologically implausible.
By improving decoy realism, we aim to strengthen FDR estimation and enhance the reliability of proteomics analyses.
