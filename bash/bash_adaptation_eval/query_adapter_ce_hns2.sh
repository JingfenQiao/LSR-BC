#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=ce_doc
#SBATCH --mem=30G
#SBATCH --time=30:00:00
#SBATCH --array=0-4%3
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --output=./log/ce_new_model_as_teacher_sp_v3_doc%a.out
#SBATCH --error=./log/ce_new_model_as_teacher_sp_v3_doc%a.out

epochs=(9 10 11 12 13)
epoch=${epochs[$SLURM_ARRAY_TASK_ID]}

path=/ivi/ilps/personal/jkang1/jf/lsr-bc/ce_new_model_as_teacher_sp_v3_doc
CUDA_VISIBLE_DEVICES=0 python bc/evaluate_beir_query.py \
  --doc_encoder naver/splade-v3-doc \
  --query_encoder $path/epoch-$epoch \
  --benchmark BEIR \
  --batch_size 32 \
  --output_folder results/ce_new_model_as_teacher_sp_v3_doc/epoch-$epoch