#!/bin/bash

script="$1"

#SBATCH --job-name=my_job
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=tue.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU

# Load modules or software if needed
# In the example PyTorch is made available for import in to my_sript.py
module load PyTorch/2.1.2-foss-2023a

# Execute the script or command
python src/main.py