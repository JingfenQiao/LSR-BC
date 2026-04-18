#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=ce
#SBATCH --mem=60G
#SBATCH --time=60:00:00
#SBATCH --output=./log/precompute_ce_hn.output
#SBATCH --error=./log/precompute_ce_hn.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request two GPUs per task

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/pre_compute_ce_hn.py \
    --triplet_path /ivi/ilps/personal/jqiao/lsr-bc/data/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz \
    --out_dir /ivi/ilps/personal/jqiao/lsr-bc/data/train_set/splade_v3_precompute_ce_hn.jsonl.gz
