#!/bin/bash

#SBATCH --job-name=mlp_random_smart_gen_%j
#SBATCH --output=mlp_random_smart_gen_%j.txt
#SBATCH --partition=mcs.gpu.q      # Choose a partition that has GPUs
#SBATCH --time=1:00:00
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

python -u src/decoy_gen.py --command evaluate --classifier mlp --target_file data/targets/UP000002311_559292.fasta --deocy_file data/decoys/UP000002311_559292.random_pos_smart_masking_esm_8M.0.fasta