#!/bin/bash
#SBATCH --job-name=icl-train
#SBATCH --account=benv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module purge
module load cuda/12.6
export PYTHONUSERBASE=/projects/benv/sdaniels2/pyuserbase
export PATH="$PYTHONUSERBASE/bin:/sw/user/python/miniforge3-pytorch-24.11.3/bin:$PATH"
export CONDA_NO_PLUGINS=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# # Use all CPUs: split across 4 ranks
# export NPROC_PER_NODE=1
# export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NPROC_PER_NODE ))
# export MKL_NUM_THREADS=${OMP_NUM_THREADS}
# export NUMEXPR_MAX_THREADS=${OMP_NUM_THREADS}
# export TORCH_NUM_THREADS=${OMP_NUM_THREADS}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

# python -c "import os; import torch; print('ENV: RANK', os.environ.get('RANK'), 'LOCAL_RANK', os.environ.get('LOCAL_RANK'), 'WORLD_SIZE', os.environ.get('WORLD_SIZE'), 'CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch.cuda.device_count()', torch.cuda.device_count());"
# python -c "import torch; print('current_device', torch.cuda.current_device()); print('allocated', torch.cuda.memory_allocated(), 'reserved', torch.cuda.memory_reserved())"

cd /projects/benv/sdaniels2/interleaved_time_series/TFs_do_KF_ICL/src

rm -f hpc_ckpt_*.ckpt
# srun --cpu-bind=cores torchrun --nproc_per_node=${NPROC_PER_NODE} --standalone data_train.py

python data_train.py

echo "Job finished at: $(date)"
