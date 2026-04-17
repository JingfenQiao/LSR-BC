#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=run
#SBATCH --mem=60G
#SBATCH --time=60:00:00
#SBATCH --output=./log/new_model_as_teacher_num_negs_30.output
#SBATCH --error=./log/new_model_as_teacher_num_negs_30.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:4   # Request two GPUs per task

export WANDB_API_KEY="2c8e7ebe2df3dfb239748abdfdf7fef9b4b4440d"

# CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run2.py \
#     --doc_encoder_old_ckpt rasyosef/splade-tiny \
#     --query_encoder_init_ckpt naver/splade-v3 \
#     --out_dir ./new_model_as_teacher \

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u query_adapter/run2.py \
    --doc_encoder_old_ckpt rasyosef/splade-tiny \
    --query_encoder_init_ckpt naver/splade-v3 \
    --out_dir ./new_model_as_teacher_num_negs_30 \
    --num_negs 30 \
    --epochs 20 \
    --batch_size 8 \