
path=model
query_adapter_checkpoint=""

CUDA_VISIBLE_DEVICES=0 python bc/evaluate_beir_query_adapter.py \
  --doc_encoder rasyosef/splade-tiny \
  --query_encoder $path/$query_adapter_checkpoint \
  --benchmark BEIR \
  --batch_size 32 \
  --output_folder results/$query_adapter_checkpoint