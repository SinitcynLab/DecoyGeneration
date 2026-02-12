#!/bin/bash

#SBATCH --job-name=eval_nov_%j
#SBATCH --output=eval_nov_%j.txt
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

python -u src/decoy_gen.py --command evaluate --classifier rnn --encoder_model protbert --target_file data/targets/UP000002311_559292.fasta --decoy_files data/decoys/UP000002311_559292.new.0.nov.fasta data/decoys/UP000002311_559292.random_replace.0.fasta --decoy_ids new_nov random_replace