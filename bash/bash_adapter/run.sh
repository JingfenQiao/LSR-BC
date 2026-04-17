#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=run
#SBATCH --mem=60G
#SBATCH --time=60:00:00
#SBATCH --output=./log/query_adapter2.output
#SBATCH --error=./log/query_adapter2.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

export WANDB_API_KEY="2c8e7ebe2df3dfb239748abdfdf7fef9b4b4440d"

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run.py \
    --doc_encoder_old_ckpt rasyosef/splade-tiny \
    --query_encoder_init_ckpt naver/splade-v3 \
    --out_dir ./splade_query_student_distill \
