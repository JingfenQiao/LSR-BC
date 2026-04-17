#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=pre_compute
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=./log/bc_pre_compute_tiny.output
#SBATCH --error=./log/bc_pre_compute_tiny.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:4   # Request four GPUs per task

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u adapter/pre_compute.py \
#   --old_model_name rasyosef/splade-tiny \
#   --new_model_name naver/splade-v3 \
#   --batch_size 128 \
#   --num_docs 1000000 \
#   --save_dir /fnwi_fs/ivi/irlab/projects/mllm_yk/jf/lsrbc/train_set/tiny


CUDA_VISIBLE_DEVICES=0,1,2,3 python -u adapter/pre_compute2.py \
  --old_model_name rasyosef/splade-tiny \
  --new_model_name naver/splade-v3 \
  --batch_size 128 \
  --num_docs 1000000 \
  --save_dir /fnwi_fs/ivi/irlab/projects/mllm_yk/jf/lsrbc/train_set/prefix_tiny