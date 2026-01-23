#!/bin/bash
#SBATCH --mem=725g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=ghx4
#SBATCH --time=48:00:00
#SBATCH --job-name=mask_back
#SBATCH --account=benv-dtai-gh
#SBATCH --output=icl-train-%x-%j.out
#SBATCH --error=icl-train-%x-%j.err
### GPU options ###
#SBATCH --gpus-per-node=4

source /projects/benv/sdaniels2/interleaved_time_series/TFs_do_KF_ICL/setup_access_gpu.sh

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python -c "import os; import torch; print('ENV: RANK', os.environ.get('RANK'), 'LOCAL_RANK', os.environ.get('LOCAL_RANK'), 'WORLD_SIZE', os.environ.get('WORLD_SIZE'), 'CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch.cuda.device_count()', torch.cuda.device_count());"
# python -c "import torch; print('current_device', torch.cuda.current_device()); print('allocated', torch.cuda.memory_allocated(), 'reserved', torch.cuda.memory_reserved())"

cd /projects/benv/sdaniels2/interleaved_time_series/TFs_do_KF_ICL/src

rm -f hpc_ckpt_*.ckpt
# srun --cpu-bind=cores torchrun --nproc_per_node=${NPROC_PER_NODE} --standalone data_train.py

time srun python data_train.py

echo "Job finished at: $(date)"
