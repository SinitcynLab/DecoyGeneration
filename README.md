# PLM-based Decoy Generation for Proteomics

## Overview

In mass spectrometry (MS)-based proteomics, peptide identification relies on matching experimental spectra with theoretical spectra derived from protein databases. To identify the rate in which false matches occur, also known as the False Discovery Rate (FDR), researcers typically use target-decoy strategies. In such regimes, synthetic decoy peptides serve as negative controls

The quality of a given target-decoy strategy relies on several factors. Importantly, decoy matches are assumed to follow the same score distribution as incorrect target matches. This assumption critically relies on the decoys biochemically resembling true proteins. If this assumption is violated, it leads to reduced confidence in the FDR-estimation.

Widely adopted decoy-generation strategies (consisting of e.g. shuffling or reversing all peptides found in a protein sequence) typically fail to yield realistic proteins because of the crude way in which they are devised. As a result, these decoys tend to follow score distributions which are vastly different from those followed by incorrect target matches. This project attempts to address this shortcoming by conceptualizing a novel approach to generating decoys. This method relies on using various Protein Language Models (PLM) to generate decoys which more closely resemble target proteins. In brief, the process consists of masking specific amino acids in a given target sequence, having the PLM fill them in with feasible alternatives, and thus obtaining a decoy.

To verify that these PLM-generated decoys more closely resemble target proteins, this work offers various machine learning models that estimate how easy it is to separate target proteins from various types of decoy. Moreover, we employ MSFragger to compare the score distributions followed by various decoy types.

Together, these two types of evidence lend strong credence to the claim that PLM-based decoys are better suited for target-decoy strategies than conventionally used methods.

## Brief Manual
This respository constitutes a first exploration of the project stipulated above. It can be entirely run for the CLI and its main entry point is src/decoy_gen.py, referenced from the root of the repository.

To obtain more information on parameters and their use, please run 'python src/decoy_gen.py --help'. The parameters of the classifiers in this project can be examined by reviewing the manuscript associated with this project at [CITATION].

Below, we provide some example uses of the codebase.

*Generate decoys from UP000002311_559292.fasta using the reverse decoy generation method*:

python src/decoy_gen.py --command generate --generators reverse --target_file data/targets/UP000002311_559292.fasta --output_directory data/decoys

*Have the MLP try to tell apart the sequences in UP000002311_559292.fasta from the sequences in UP000002311_559292.reverse.fasta*:

python src/decoy_gen.py --command classify --classifier mlp --target_file data/targets/UP000002311_559292.fasta --decoy_files data/decoys/UP000002311_559292.reverse.fasta --decoy_ids reverse

*Time how quickly the generation methods reverse, esm8M_32bit and max_prob_smart_esm are able to convert the first 200 sequences in UP000002311_559292.fasta*:

python src/decoy_gen.py --command time --generators reverse esm8M_32bit max_prob_smart_esm --target_file data/targets/UP000002311_559292.fasta --timing_sample 200