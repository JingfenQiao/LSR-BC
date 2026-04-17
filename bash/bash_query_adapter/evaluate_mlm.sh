#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=tiny
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --output=./log/ce_new_model_as_teacher_mse%a.out
#SBATCH --error=./log/ce_new_model_as_teacher_mse%a.out

epochs=(9 10 11 12)
epoch=${epochs[$SLURM_ARRAY_TASK_ID]}
path=/ivi/ilps/personal/jqiao/lsr-bc/query_adapter_model/ce_new_model_as_teacher_mse

CUDA_VISIBLE_DEVICES=0 python -u bc/evaluate_beir.py \
  --doc_encoder rasyosef/splade-tiny \
  --query_encoder $path/epoch-$epoch \
  --benchmark BEIR \
  --batch_size 32 \
  --output_folder results/ce_new_model_as_teacher_mse/epoch-$epoch

