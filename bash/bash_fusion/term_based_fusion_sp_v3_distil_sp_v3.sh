#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=sp_v3_distil
#SBATCH --mem=30G
#SBATCH --time=120:00:00
#SBATCH --output=./log/new_sp_v3_old_sp_v3_distil_fusion_type_%a.output
#SBATCH --error=./log/new_sp_v3_old_sp_v3_distil_fusion_type_%a.output
#SBATCH --array=1-2   # We have 4 files
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

old_model=(naver/splade-v3-distilbert sp_v3_distil)
new_model=(naver/splade-v3 sp_v3)

type=(2 5)
fusion_type=${type[$SLURM_ARRAY_TASK_ID-1]}

CUDA_VISIBLE_DEVICES=0 python bc/evaluate_beir_fusion.py \
  --old_model ${old_model[0]} \
  --new_model ${new_model[0]} \
  --fusion_type $fusion_type \
  --batch_size 256 \
  --benchmark BEIR \
  --output_folder fusion_results/new_${new_model[1]}_old_${old_model[1]}_fusion_type_$fusion_type
