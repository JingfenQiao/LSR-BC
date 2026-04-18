#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=ce
#SBATCH --mem=60G
#SBATCH --time=90:00:00
#SBATCH --output=./log/pre_compute_or_hn.output
#SBATCH --error=./log/pre_compute_or_hn.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:2   # Request three GPUs per task

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/pre_compute_or_hn.py \
    --out_dir /ivi/ilps/personal/jqiao/lsr-bc/data/train_set/splade_v3_precompute_openresearch_hn.jsonl.gz
