# PLM-based Decoy Generation for Proteomics

## Overview

In mass spectrometry (MS)-based proteomics, peptide identification relies on matching experimental spectra with theoretical spectra derived from protein databases. To identify the rate in which false matches occur, also known as the False Discovery Rate (FDR), researcers typically use *target-decoy* strategies. In such regimes, synthetic *decoy peptides* serve as negative controls

The quality of a given target-decoy strategy relies on several factors. Importantly, the decoy matches are assumed to follow the same *score distribution* as the incorrect target matches. This second assumption critically relies on the decoy peptides biochemically resembling true proteins. If this assumption is violated, then this leads to reduced confidence in the FDR-estimation.

Most widely adopted decoy-generation strategies (consisting of e.g. shuffling or reversing all peptides found in a protein sequence) typically fail to resemble realistic proteins because of the crude way in which they are devised. As a result, these simplistic decoys tend to follow score distributions which are vastly different from those followed by false target matches. In this project, we attempt to address this shortcoming of existing methods by conceptualizing a novel approach to generating decoys. This new method relies on using various Protein Language Models (PLM) to generate decoys which more closely resemble target proteins. In brief, the process consists of masking specific amino acids in a given target sequence, having the PLM fill them in with feasible alternatives, and thus obtaining a decoy.

To verify that these PLM-generated decoys more closely resemble target proteins, this project offers several machine learning classifiers which are trained to tell target and decoy peptides apart. By comparing the performance of these classifiers on datasets consisting of target proteins and various decoy types, we estimate how well each decoy type is separable from target proteins. If a trained classifier has difficulty telling targets and decoys apart under a given decoy-generation strategy, we take this as evidence that the decoys statistically resemble the target proteins. Additionally, we present a means to use the MSFragger database for estimating the score distributions of various decoy types and compare them to the score distributions of target peptides. Accordingly, we can verify whether the score distributions of PLM-generated decoys indeed more closely resemble those of true target peptides.

Together, these two types of evidence lend strong credence to the claim that PLM-based decoys are better suited for target-decoy strategies than conventionally used methods.

## Manual
This respository constitutes a first exploration of the 