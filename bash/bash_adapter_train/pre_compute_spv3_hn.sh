#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=train
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=./log/pre_compute_hn.output
#SBATCH --error=./log/pre_compute_hn.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request two GPUs per task

CUDA_VISIBLE_DEVICES=0 python -u query_adapter/pre_compute_hn.py \
    --triplet_path /ivi/ilps/personal/jqiao/lsr-bc/data/run.msmarco-v1-passage.train.splade-v3.txt \
    --out_dir /ivi/ilps/personal/jqiao/lsr-bc/data/train_set/splade_v3_precompute_splade_v3_hn.jsonl.gz
