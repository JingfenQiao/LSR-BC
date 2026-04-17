#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=tiny
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=./log/run.output
#SBATCH --error=./log/run.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run.py \
    --query_encoder_init_ckpt naver/splade-v3 \
    --doc_encoder_old_ckpt rasyosef/splade-tiny \
    --batch_size 8 \
    --out_dir query_adapter_model/ce_new_model_as_teacher_mse/ \
    --triplet_path /ivi/ilps/personal/jqiao/lsr-bc/data/train_set/splade_v3_precompute_ce_hn.jsonl.gz \
    --num_negs 20 \
    --epochs 15
