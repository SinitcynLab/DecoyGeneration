#!/bin/bash

#SBATCH --job-name=svm_umap_%j
#SBATCH --output=svm_umap_%j.txt
#SBATCH --partition=tue.default.q      # Choose a partition that has GPUs
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G

module load Python/3.11.3-GCCcore-12.3.0
module load Anaconda3/2023.09-0
eval "$(conda shell.bash activate)"
source activate decoy_gen

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python -u src/run/cross_val_svm_umap_kernel.py