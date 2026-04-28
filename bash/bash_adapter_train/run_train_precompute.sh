

doc_model=naver/splade-v3-distilbert
query_model=naver/splade-v3

CUDA_VISIBLE_DEVICES=0,1 python -u query_adapter/run_precompute_ce.py \
    --doc_encoder_old_ckpt $doc_model \
    --query_encoder_init_ckpt $query_model \
    --data_path data/train_set/splade_v3_precompute_ce_hn.jsonl.gz \
    --out_dir ./ce_new_model_as_teacher_sp_v3_distil \
    --num_negs 20 \
    --epochs 15