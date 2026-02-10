#!/bin/bash

#SBATCH --job-name=gen_all_ml_%j
#SBATCH --output=gen_all_ml_%j.txt
#SBATCH --partition=mcs.gpu.q      # Choose a partition that has GPUs
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1 

module load Python/3.11.3-GCCcore-12.3.0
module load Anaconda3/2023.09-0
eval "$(conda shell.bash activate)"
source activate decoy_gen

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python -u src/decoy_gen.py --command evaluate --classifier mlp --encoder_model esm --target_file data/targets/UP000002311_559292.fasta --decoy_files data/decoys/UP000002311_559292.protbert.best.c1.32b.0.fasta --decoy_ids protbert_32bit
python -u src/decoy_gen.py --command evaluate --classifier mlp --encoder_model protbert --target_file data/targets/UP000002311_559292.fasta --decoy_files data/decoys/UP000002311_559292.esm35M.best.c1.32b.0.fasta data/decoys/UP000002311_559292.esm150M.best.c1.32b.0.fasta --decoy_ids esm35M_32bit esm150M_32bit
python -u src/decoy_gen.py --command evaluate --classifier mlp --encoder_model protbert --target_file data/targets/UP000002311_559292.fasta --decoy_files data/decoys/UP000002311_559292.esm8M.best.c1.16b.0.fasta data/decoys/UP000002311_559292.esm35M.best.c1.16b.0.fasta data/decoys/UP000002311_559292.esm150M.best.c1.16b.0.fasta data/decoys/UP000002311_559292.esm650M.best.c1.16b.0.fasta data/decoys/UP000002311_559292.esm3B.best.c1.16b.0.fasta --decoy_ids esm8M_16bit esm35M_16bit esm150M_16bit esm650M_16bit esm3B_16bit