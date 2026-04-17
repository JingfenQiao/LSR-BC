#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=fusion
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=./log/adapter_new_sp_v3_old_sp_tiny_fusion_type_%a.output
#SBATCH --error=./log/adapter_new_sp_v3_old_sp_tiny_fusion_type_%a.output
#SBATCH --array=1-2   # We have 2 files
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

old_model=(rasyosef/splade-tiny sp_tiny)
new_model=(naver/splade-v3 sp_v3)
adapter_model=(/ivi/ilps/personal/jkang1/jf/lsr-bc/ce_new_model_as_teacher/epoch-11 ce_new_model_as_teacher)
type=(5 7)

fusion_type=${type[$SLURM_ARRAY_TASK_ID-1]}

CUDA_VISIBLE_DEVICES=0 python bc/evaluate_beir_fusion_adapter.py \
  --old_model ${old_model[0]} \
  --new_model ${new_model[0]} \
  --adapter_model ${adapter_model[0]} \
  --fusion_type $fusion_type \
  --batch_size 256 \
  --benchmark BEIR \
  --output_folder fusion_results/new_${new_model[1]}_old_${old_model[1]}_fusion_type_$fusion_type
