#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=query_adapter
#SBATCH --mem=30G
#SBATCH --time=30:00:00
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --output=./log/ce_new_model_as_teacher%a.out
#SBATCH --error=./log/ce_new_model_as_teacher%a.out

epochs=(12 13 14)
epoch=${epochs[$SLURM_ARRAY_TASK_ID]}

path=/ivi/ilps/personal/jkang1/jf/lsr-bc/ce_new_model_as_teacher
CUDA_VISIBLE_DEVICES=0 python bc/evaluate_beir_query.py \
  --doc_encoder rasyosef/splade-tiny \
  --query_encoder $path/epoch-$epoch \
  --benchmark BEIR \
  --batch_size 32 \
  --output_folder results/ce_new_model_as_teacher/epoch-$epoch