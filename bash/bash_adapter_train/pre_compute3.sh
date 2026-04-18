#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=pre_compute
#SBATCH --mem=30G
#SBATCH --time=72:00:00                  # Increased (v3 is slow)
#SBATCH --output=./log/pre_compute_%a.out   # Separate logs for each task
#SBATCH --error=./log/pre_compute_%a.out
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --array=0-1


if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    MODEL="rasyosef/splade-tiny"
    BATCH=64
    DIR="logit_tiny"
else
    MODEL="naver/splade-v3"
    BATCH=64
    DIR="logit_v3"
fi

SAVE_DIR="/fnwi_fs/ivi/irlab/projects/mllm_yk/jf/lsrbc/train_set/$DIR"
mkdir -p $SAVE_DIR


CUDA_VISIBLE_DEVICES=0,1 python doc_adapter/pre_compute4.py \
    --model_name $MODEL \
    --batch_size $BATCH \
    --num_docs 1000000 \
    --save_dir $SAVE_DIR \
    --shard_size 5000