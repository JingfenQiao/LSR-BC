#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=q_op_doc_v3_distill_d_sp_v3
#SBATCH --mem=30G
#SBATCH --time=60:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/lsr-bc/log/q_op_doc_v3_distill_d_sp_v3.output
#SBATCH --error=/ivi/ilps/personal/jqiao/lsr-bc/log/q_op_doc_v3_distill_d_sp_v3.output
#SBATCH --array=1   # We have 5 files
#SBATCH --gres=gpu:nvidia_rtx_a6000   # Request one GPU per task

python bc/evaluate_beir_asy.py \
  --query_encoder opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill \
  --doc_encoder naver/splade-v3 \
  --benchmark BEIR \
  --batch_size 512 \
  --output_folder results/q_op_doc_v3_distill_d_sp_v3
