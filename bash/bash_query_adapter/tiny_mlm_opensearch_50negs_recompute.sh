#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=tiny
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=./log/mlm_opensearch_40negs_recompute.output
#SBATCH --error=./log/mlm_opensearch_40negs_recompute.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run.py \
    --batch_size 8 \
    --out_dir query_adapter_model/mlm_opensearch_40negs_recompute/ \
    --num_negs 40 \
    --epochs 20
