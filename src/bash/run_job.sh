#!/bin/bash

#SBATCH --job-name=my_job
#SBATCH --output=cv_%j.txt
#SBATCH --partition=tue.default.q         # Choose a partition that has GPUs
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=6G

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python -u src/test/cross_val_test.py