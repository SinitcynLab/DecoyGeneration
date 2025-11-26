#!/bin/bash

#SBATCH --job-name=my_job
#SBATCH --output=gen_s_%j.txt
#SBATCH --partition=tue.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python -u src/main.py