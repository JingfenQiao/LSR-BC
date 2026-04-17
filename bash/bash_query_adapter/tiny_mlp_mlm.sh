#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=tiny
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=./log/run_mlp_mlm.output
#SBATCH --error=./log/run_mlp_mlm.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run_mlp_mlm.py \
    --batch_size 8
