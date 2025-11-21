#!/bin/bash
#SBATCH --job-name=test_basic
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1          # 1 GPU

echo "Test started"
hostname
pwd
echo "Test finished"