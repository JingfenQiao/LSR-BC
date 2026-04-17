#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=evaluation
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr-bc/log/q_op_v2_distill_d_op_v2_distill.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr-bc/log/q_op_v2_distill_d_op_v2_distill.output
#SBATCH --array=1   # We have 5 files
#SBATCH --gres=gpu:nvidia_rtx_a6000   # Request one GPU per task

python bc/evaluate_beir.py \
  --query_encoder opensearch-project/opensearch-neural-sparse-encoding-v2-distill \
  --doc_encoder opensearch-project/opensearch-neural-sparse-encoding-v2-distill \
  --benchmark BEIR \
  --batch_size 256 \
  --output_folder results/q_op_v2_distill_d_op_v2_distill


