#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=bm25
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=./log/pre_compute_bm25.output
#SBATCH --error=./log/pre_compute_bm25.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request two GPUs per task

CUDA_VISIBLE_DEVICES=0 python -u query_adapter/pre_compute_bm25.py
