#!/bin/bash

#SBATCH --job-name=rnn_spectra_shuffle_esm650_%j
#SBATCH --output=rnn_spectra_shuffle_esm650_%j.txt
#SBATCH --partition=tue.gpu.q      # Choose a partition that has GPUs
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

python -u src/run/cross_val_rnn_spectra.py