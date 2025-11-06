#!/bin/bash
#SBATCH --job-name=icl-train
#SBATCH --account=benv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module purge
module load cuda/12.6
export PYTHONUSERBASE=/projects/benv/sdaniels2/pyuserbase
export PATH="$PYTHONUSERBASE/bin:/sw/user/python/miniforge3-pytorch-24.11.3/bin:$PATH"
export CONDA_NO_PLUGINS=1

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

cd /projects/benv/sdaniels2/interleaved_time_series/TFs_do_KF_ICL/src
python data_train.py --resume_train --model_name ortho_haar_big_unmask_backstory_no_leak_mid

echo "Job finished at: $(date)"
