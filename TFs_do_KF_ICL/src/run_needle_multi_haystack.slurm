#!/bin/bash
#SBATCH --job-name=needle_inference
#SBATCH --gres=gpu:1          # 1 GPU
#SBATCH --cpus-per-task=32    # 32 CPUs per GPU (optimal)
#SBATCH --mem=200G            # ~200GB per GPU
#SBATCH --output=slurm-%x_%j.out # Standard output file
#SBATCH --error=slurm-%x_%j.err  # Standard error file

# Debug: Print environment info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Load modules if needed
# module load cuda/11.8
# module load python/3.9

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Run with error tracing
set -e  # Exit on error
set -x  # Print commands

# Your GPU code
python data_train.py --make_preds --multi_haystack --only_needle_pos --hard_coded_ckpt 122000 --model_name ortho_haar_medium_single_gpu

echo "Job finished at: $(date)"