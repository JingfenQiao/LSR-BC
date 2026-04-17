#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=ce_lexical
#SBATCH --mem=60G
#SBATCH --time=60:00:00
#SBATCH --output=./log/ce_new_model_as_teacher_sp_v3_lexical.output
#SBATCH --error=./log/ce_new_model_as_teacher_sp_v3_lexical.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

export WANDB_API_KEY="2c8e7ebe2df3dfb239748abdfdf7fef9b4b4440d"

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run_ce_new_model.py \
    --doc_encoder_old_ckpt naver/splade-v3-lexical \
    --query_encoder_init_ckpt naver/splade-v3 \
    --out_dir ./ce_new_model_as_teacher_sp_v3_lexical \
    --num_negs 20 \
    --epochs 20

