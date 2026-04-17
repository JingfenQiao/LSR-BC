#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=ce_lexical
#SBATCH --mem=30G
#SBATCH --time=30:00:00
#SBATCH --array=0
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --output=./log/ce_new_model_as_teacher_sp_v3_lexical%a.out
#SBATCH --error=./log/ce_new_model_as_teacher_sp_v3_lexical%a.out

epochs=(12)
epoch=${epochs[$SLURM_ARRAY_TASK_ID]}

path=/ivi/ilps/personal/jkang1/jf/lsr-bc/ce_new_model_as_teacher_sp_v3_lexical
CUDA_VISIBLE_DEVICES=0 python bc/evaluate_beir_query.py \
  --doc_encoder naver/splade-v3-lexical \
  --query_encoder $path/epoch-$epoch \
  --benchmark BEIR \
  --batch_size 32 \
  --output_folder results/ce_new_model_as_teacher_sp_v3_lexical/epoch-$epoch