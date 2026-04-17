#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=spv3_doc
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=./log/mlm_opensearch_20negs_recompute_sp_v3_doc.output
#SBATCH --error=./log/mlm_opensearch_20negs_recompute_sp_v3_doc.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run.py \
    --query_encoder_init_ckpt naver/splade-v3 \
    --doc_encoder_old_ckpt naver/splade-v3-doc \
    --batch_size 8 \
    --out_dir query_adapter_model/mlm_opensearch_20negs_recompute_sp_v3_doc/ \
    --num_negs 20 \
    --epochs 20
