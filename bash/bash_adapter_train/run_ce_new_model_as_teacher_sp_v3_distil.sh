#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=ce_distil
#SBATCH --mem=60G
#SBATCH --time=60:00:00
#SBATCH --output=./log/ce_new_model_as_teacher_sp_v3_distil.output
#SBATCH --error=./log/ce_new_model_as_teacher_sp_v3_distil.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

export WANDB_API_KEY="2c8e7ebe2df3dfb239748abdfdf7fef9b4b4440d"

doc_model=naver/splade-v3-distilbert
query_model=naver/splade-v3

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run_ce_new_model.py \
    --doc_encoder_old_ckpt $doc_model \
    --query_encoder_init_ckpt $query_model \
    --out_dir ./ce_new_model_as_teacher_sp_v3_distil \
    --num_negs 20 \
    --epochs 15