
CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/pre_compute_ce_hn.py \
    --triplet_path data/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz \
    --out_dir data/train_set/splade_v3_precompute_ce_hn.jsonl.gz