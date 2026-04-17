#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=q_sp_v3_d_sp_v3_lexical
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr-bc/log/q_sp_v3_d_sp_v3_lexical.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr-bc/log/q_sp_v3_d_sp_v3_lexical.output
#SBATCH --array=1   # We have 5 files
#SBATCH --gres=gpu:nvidia_rtx_a6000   # Request one GPU per task

python bc/evaluate_beir_asy.py \
  --query_encoder naver/splade-v3 \
  --doc_encoder naver/splade-v3-lexical \
  --benchmark BEIR \
  --batch_size 512 \
  --output_folder results/q_sp_v3_d_sp_v3_lexical