#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=op_multilingual_v1_d_sp_v3
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr-bc/log/op_multilingual_v1_d_sp_v3.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr-bc/log/op_multilingual_v1_d_sp_v3.output
#SBATCH --array=1   # We have 5 files
#SBATCH --gres=gpu:nvidia_rtx_a6000   # Request one GPU per task


python bc/evaluate_beir.py \
  --query_encoder opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 \
  --doc_encoder naver/splade-v3 \
  --benchmark BEIR \
  --batch_size 512 \
  --output_folder results/op_multilingual_v1_d_sp_v3


