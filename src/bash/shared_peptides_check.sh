#!/bin/bash

#SBATCH --job-name=shared_peptides_check_%j
#SBATCH --output=shared_peptides_check_%j.txt
#SBATCH --partition=tue.default.q      # Choose a partition that has GPUs
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=6G

module load Python/3.11.3-GCCcore-12.3.0
module load Anaconda3/2023.09-0
eval "$(conda shell.bash activate)"
source activate decoy_gen

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python -u src/io/shared_peptides_checker.py -s data/decoys/UP000002311_559292.esm650M.best.c1.32b.0.fasta 
python -u src/io/shared_peptides_checker.py -s data/targets/UP000002311_559292.fasta
python -u src/io/shared_peptides_checker.py -s data/decoys/UP000002311_559292.esm8M.best.c1.32b.0.fasta
python -u src/io/shared_peptides_checker.py -s data/decoys/UP000002311_559292.shuffle.0.fasta
python -u src/io/shared_peptides_checker.py -s data/decoys/UP000002311_559292.reverse.fasta 
python -u src/io/shared_peptides_checker.py -s data/decoys/UP000002311_559292.diann.fasta 
