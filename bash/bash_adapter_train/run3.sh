#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=run
#SBATCH --mem=60G
#SBATCH --time=60:00:00
#SBATCH --output=./log/ce_as_teacher.output
#SBATCH --error=./log/ce_as_teacher.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

export WANDB_API_KEY="2c8e7ebe2df3dfb239748abdfdf7fef9b4b4440d"
CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run_distill_ce.py \
    --doc_encoder_old_ckpt rasyosef/splade-tiny \
    --query_encoder_init_ckpt naver/splade-v3 \
    --out_dir ./ce_as_teacher