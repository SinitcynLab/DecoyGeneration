#!/bin/bash

#SBATCH --job-name=umap_gen_6000_%j
#SBATCH --output=umap_gen_6000_%j.txt
#SBATCH --partition=tue.gpu.q      # Choose a partition that has GPUs
#SBATCH --time=10:00:00
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

python -u src/visualization/umap_visualizer.py --files data/targets/UP000002311_559292.fasta data/decoys/UP000002311_559292.shuffle.0.fasta data/decoys/UP000002311_559292.esm650M.best.c1.0.fasta --identifiers "Target" "Shuffle" "ESM 650" -n 6059
python -u src/visualization/umap_visualizer.py --files data/targets/UP000002311_559292.fasta data/decoys/UP000002311_559292.reverse.fasta data/decoys/UP000002311_559292.esm650M.best.c1.0.fasta --identifiers "Target" "Reverse" "ESM 650" -n 6059
python -u src/visualization/umap_visualizer.py --files data/targets/UP000002311_559292.fasta data/decoys/UP000002311_559292.diann.fasta data/decoys/UP000002311_559292.esm650M.best.c1.0.fasta --identifiers "Target" "DIA-NN" "ESM 650" -n 6059