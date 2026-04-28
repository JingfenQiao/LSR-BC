
#example
BASELINES=(baseline6_1 baseline6_2 baseline6_3 baseline6_4 baseline23_1 baseline23_2 baseline23_3 baseline23_4)
STREAM=1

python -u bc/streaming/search_streaming_new.py \
  --baselines "$BASELINES" \
  --stream "$STREAM" \
  --skip_encoding \
  --metadata_path "$REPO_DIR/metadata_streaming_splade-tiny_splade-v3.json" \
  --batch_size 1000 \
  --top_k 100 \
    --output_dir "$OUTPUT_ROOT" \
  --old_model old_v3distilbert \
  --new_model new
