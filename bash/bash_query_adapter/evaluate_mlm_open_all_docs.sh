#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=tiny
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --array=0-6
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --output=./log/mlm_all_docs%a.out
#SBATCH --error=./log/mlm_all_docs%a.out

epochs=(7 8 9 10)
epoch=${epochs[$SLURM_ARRAY_TASK_ID]}

path=/ivi/ilps/personal/jqiao/lsr-bc/query_adapter_model/mlm_all_docs_40negs_kl_mse_all_docs


CUDA_VISIBLE_DEVICES=0 python -u bc/evaluate_beir.py \
  --doc_encoder rasyosef/splade-tiny \
  --query_encoder $path/epoch-$epoch \
  --benchmark BEIR \
  --batch_size 32 \
  --output_folder results/mlm_all_docs_40negs_kl_mse_all_docs/epoch-$epoch