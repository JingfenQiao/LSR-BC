#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=or_20negs_recompute
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --array=0-4
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --output=./log/mlm_opensearch_20negs_recompute%a.out
#SBATCH --error=./log/mlm_opensearch_20negs_recompute%a.out

epochs=(10 11 12 13 14)

epoch=${epochs[$SLURM_ARRAY_TASK_ID]}
path=/ivi/ilps/personal/jqiao/lsr-bc/query_adapter_model/mlm_opensearch_20negs_recompute


CUDA_VISIBLE_DEVICES=0 python -u bc/evaluate_beir.py \
  --doc_encoder rasyosef/splade-tiny \
  --query_encoder $path/epoch-$epoch \
  --benchmark BEIR \
  --batch_size 64 \
  --output_folder results/mlm_opensearch_20negs_recompute/epoch-$epoch

# /ivi/ilps/personal/jqiao/lsr-bc/query_adapter_model/mlm_all_docs_50negs_kl_mse
