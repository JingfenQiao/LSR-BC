# LSR-BC: Backwards-Compatible Learned Sparse Retrieval

LSR-BC studies how to adapt a new SPLADE query encoder to remain compatible with a **frozen legacy document index**, avoiding full re-indexing while preserving retrieval quality.

---

## Environment

```bash
pip install -r requirements.txt
```

---

## Model Checkpoints

All checkpoints are loaded from Hugging Face Hub automatically.

| Model | HF Hub ID | Role |
|---|---|---|
| SPLADE v3 | `naver/splade-v3` | New query encoder (M_new) |
| SPLADE Tiny | `rasyosef/splade-tiny` | Old doc encoder |
| SPLADE Tiny Adapter | `JFJFJFen/splade_v3_doc_query_adapter` | QAdapter for SPLADE Tiny index |
| SPLADE v3 Doc | `naver/splade-v3-doc` | Old doc encoder |
| SPLADE v3 Doc Adapter | `naver/splade-cocondenser-selfdistil` | QAdapter for SPLADE v3 Doc index |
| SPLADE v3 Lexical | `naver/splade-v3-lexical` | Old doc encoder |
| SPLADE v3 Lexical Adapter | `JFJFJFen/splade_v3_lexical_query_adapter` | QAdapter for SPLADE v3 Lexical index |

---

## Project Structure

```
LSR-BC/
├── bc/
│   ├── query_adapter/                      # QAdapter training pipeline
│   │   ├── modelling.py                    # SpladeSparseEncoder model definition
│   │   ├── dataloader.py                   # Dataset classes and data loading utilities
│   │   ├── pre_compute_ce_hn.py            # Offline precomputation of teacher (M_new) scores
│   │   ├── run_precomupte_ce.py            # Train QAdapter using precomputed teacher scores
│   │   └── run_ce_new_model.py             # Train QAdapter with online teacher (no precompute)
│   ├── static_evaluation/                  # Offline BEIR benchmark evaluation
│   │   ├── evaluate_beir_asy.py            # Asymmetric (query ≠ doc encoder) baseline
│   │   ├── evaluate_beir_query_adapter.py  # QAdapter evaluation on BEIR
│   │   ├── evaluate_beir_rep_fusion.py     # Representation fusion evaluation
│   │   └── evaluate_beir_rank_fusion.py    # Rank fusion evaluation
│   └── stream_evaluation/                  # Streaming / online evaluation notebooks
├── bash/
│   ├── bash_adapter_train/                 # Scripts for QAdapter training
│   ├── bash_adaptation_eval/               # Scripts for adaptation evaluation
│   ├── bash_static_eval/                   # Scripts for static BEIR evaluation
│   └── bash_stream_eval/                   # Scripts for streaming evaluation
└── data/                                   # Data directory (create manually)
    └── train_set/                          # Output directory for precomputed scores
```

---

## Training

### Data

Download the MS MARCO hard negatives training file:

```bash
# Cross-encoder scores from sentence-transformers
wget https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz \
    -P data/

# Or use precomputed teacher scores (skip Step 1 below)
wget https://huggingface.co/datasets/JFJFJFen/splade_v3_teacher_scores/resolve/main/splade_v3_precompute_ce_hn.jsonl.gz \
    -P data/train_set/
```

### Step 1: Precompute Teacher Scores (optional)

Encode queries and documents with M_new and save dot-product scores as training targets. Skip this step if using the precomputed dataset above.

```bash
python bc/query_adapter/pre_compute_ce_hn.py \
    --query_encoder_init_ckpt naver/splade-v3 \
    --doc_encoder_old_ckpt naver/splade-v3-distilbert \
    --triplet_path data/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz \
    --out_dir data/train_set/splade_v3_precompute_ce_hn.jsonl.gz \
    --num_negs 20 \
    --batch_size 64
```

### Step 2a: Train QAdapter (precomputed scores)
、
```bash
python bc/query_adapter/run_precomupte_ce.py \
    --doc_encoder_old_ckpt rasyosef/splade-tiny \
    --query_encoder_init_ckpt naver/splade-v3 \
    --data_path data/train_set/splade_v3_precompute_ce_hn.jsonl.gz \
    --out_dir ./checkpoints/qadapter_tiny \
    --num_negs 20 \
    --epochs 15 \
    --batch_size 8 \
    --lr 2e-5
```

### Step 2b: Train QAdapter (online teacher)

Teacher scores are computed on-the-fly by a frozen copy of M_new. Slower but requires no precomputation.

```bash
python bc/query_adapter/run_ce_new_model.py \
    --doc_encoder_old_ckpt rasyosef/splade-tiny \
    --query_encoder_init_ckpt naver/splade-v3 \
    --out_dir ./checkpoints/qadapter_online \
    --num_negs 20 \
    --epochs 15 \
    --batch_size 8 \
    --lr 2e-5
```

---

## Evaluation

All evaluation scripts run on [BEIR](https://github.com/beir-cellar/beir) benchmarks.

### QAdapter evaluation

Evaluate a trained QAdapter against the old document index:

```bash
python bc/static_evaluation/evaluate_beir_query_adapter.py \
    --doc_encoder rasyosef/splade-tiny \
    --query_encoder ./checkpoints/qadapter_tiny/epoch-15 \
    --benchmark BEIR \
    --batch_size 32 \
    --output_folder results/qadapter_tiny
```

### Asymmetric baseline

Evaluate with mismatched query/doc encoders (no adapter), to measure the compatibility gap:

```bash
python bc/static_evaluation/evaluate_beir_asy.py \
    --query_encoder naver/splade-v3 \
    --doc_encoder naver/splade-v3-distilbert \
    --benchmark BEIR \
    --batch_size 512 \
    --output_folder results/asy_q_sp_v3_d_sp_v3_distil
```

### Representation fusion

Fuse query representations from old and new encoders (4 fusion strategies):

```bash
python bc/static_evaluation/evaluate_beir_rep_fusion.py \
    --old_model rasyosef/splade-tiny \
    --new_model naver/splade-v3 \
    --fusion_type 2 \
    --benchmark BEIR \
    --batch_size 256 \
    --output_folder results/rep_fusion_sp_v3_tiny
```

### Rank fusion

Combine ranked lists from old and new models via reciprocal rank fusion:

```bash
python bc/static_evaluation/evaluate_beir_rank_fusion.py \
    --old_query_model_name sp_tiny \
    --old_doc_model_name sp_tiny \
    --new_query_model_name sp_v3 \
    --new_doc_model_name sp_v3
```

---

## Streaming Evaluation

Evaluates backward-compatibility in a simulated streaming setting (LOTTE streams, 3 streams × 5 sessions). See [bc/streaming/docs/README.md](bc/streaming/docs/README.md) for the full workflow and dataset description. 

```bash
python bc/streaming/search_streaming_new.py \
    --baselines baseline6_1 baseline6_2 baseline23_1 baseline23_2 \
    --stream 1 \
    --skip_encoding \
    --metadata_path metadata_streaming_splade.json \
    --batch_size 1000 \
    --top_k 100 \
    --output_dir results/streaming \
    --old_model old \
    --new_model new
```
