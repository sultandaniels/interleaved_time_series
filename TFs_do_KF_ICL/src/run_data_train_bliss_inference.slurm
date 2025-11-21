#!/bin/bash
#SBATCH --job-name=2dim_bliss_inference
#SBATCH --gres=gpu:1          # 1 GPU
#SBATCH --cpus-per-task=32    # 32 CPUs per GPU (optimal)
#SBATCH --mem=200G            # ~200GB per GPU

# Your GPU code
python data_train.py --make_preds --multi_haystack --all_steps_in --model_name ortho_haar_medium_single_gpu --identical_haystack --haystack_len 2
# python data_train.py --make_preds --multi_haystack --all_steps_in --model_name ortho_haar_med_2dim