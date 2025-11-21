#!/bin/bash
#SBATCH --job-name=2dim-medium
#SBATCH --gres=gpu:1          # 1 GPU
#SBATCH --cpus-per-task=32    # 32 CPUs per GPU (optimal)
#SBATCH --mem=200G            # ~200GB per GPU

# Your GPU code
python data_train.py