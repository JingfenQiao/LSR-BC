#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=evaluation
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr-bc/log/q_sp_co_selfdistil_d_sp_co_selfdistil.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr-bc/log/q_sp_co_selfdistil_d_sp_co_selfdistil.output
#SBATCH --array=1   # We have 5 files
#SBATCH --gres=gpu:nvidia_rtx_a6000   # Request one GPU per task

python bc/evaluate_beir.py \
  --query_encoder naver/splade-cocondenser-selfdistil \
  --doc_encoder naver/splade-cocondenser-selfdistil \
  --benchmark BEIR \
  --batch_size 512 \
  --output_folder results/q_sp_co_selfdistil_d_sp_co_selfdistil


