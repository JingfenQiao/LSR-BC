#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=tiny
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=./log/run_opensearch_50negs_kl_mse.output
#SBATCH --error=./log/run_opensearch_50negs_kl_mse.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run_opensearch.py \
    --batch_size 8 \
    --out_dir query_adapter_model/mlm_all_docs_50negs_kl_mse/ \
    --num_negs 50 \
    --epochs 20
