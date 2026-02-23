#!/bin/bash

#SBATCH --job-name=mlp_esm_NC_%j
#SBATCH --output=mlp_esm_NC_%j.txt
#SBATCH --partition=mcs.gpu.q      # Choose a partition that has GPUs
#SBATCH --time=24:00:00
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

python src/decoy_gen.py --command evaluate --classifier mlp --target_file data/targets/UP000002311_559292.fasta --decoy_files data/decoys/UP000002311_559292.NC_terminus_facebook_esm2_t33_650M_UR50D.best.c1.f32.0.fasta --decoy_ids esm_NC