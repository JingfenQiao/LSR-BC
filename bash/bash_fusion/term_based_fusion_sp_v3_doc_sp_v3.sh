#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=fusion
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=./log/new_sp_v3_old_sp_v3_doc_fusion_type_%a.output
#SBATCH --error=./log/new_sp_v3_old_sp_v3_doc_fusion_type_%a.output
#SBATCH --array=1-4   # We have 4 files
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

old_model=(naver/splade-v3-doc sp_v3_doc)
new_model=(naver/splade-v3 sp_v3)

fusion_type=$SLURM_ARRAY_TASK_ID

CUDA_VISIBLE_DEVICES=0 python bc/evaluate_beir_fusion.py \
  --old_model ${old_model[0]} \
  --new_model ${new_model[0]} \
  --fusion_type $fusion_type \
  --batch_size 256 \
  --benchmark BEIR \
  --output_folder fusion_results/new_${new_model[1]}_old_${old_model[1]}_fusion_type_$fusion_type
