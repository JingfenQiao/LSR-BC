#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=q_op_v2_distill_d_sp_v3
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr-bc/log/q_op_v2_distill_d_sp_v3.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr-bc/log/q_op_v2_distill_d_sp_v3.output
#SBATCH --array=1   # We have 5 files
#SBATCH --gres=gpu:nvidia_l40   # Request one GPU per task

# python bc/evaluate_beir.py \
#   --query_encoder opensearch-project/opensearch-neural-sparse-encoding-v2-distill \
#   --doc_encoder naver/splade-v3 \
#   --benchmark BEIR \
#   --batch_size 512 \
#   --output_folder results/q_op_v2_distill_d_sp_v3

python bc/evaluate_beir.py \
  --query_encoder opensearch-project/opensearch-neural-sparse-encoding-v2-distill \
  --doc_encoder naver/splade-v3 \
  --benchmark BEIR \
  --batch_size 512 \
  --tasks MSMARCO \
  --output_folder results/q_op_v2_distill_d_sp_v3