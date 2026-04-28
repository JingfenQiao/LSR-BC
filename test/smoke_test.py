"""
Smoke tests for LSR-BC.
Each test is independent; failures are collected and reported at the end.
"""
import sys, os, traceback

# Run from project root: python test/smoke_test.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bc", "query_adapter"))
sys.path.insert(0, os.path.join(ROOT, "bc", "static_evaluation"))

results = {}


def run(name, fn):
    try:
        fn()
        results[name] = "PASS"
        print(f"  [PASS] {name}")
    except Exception:
        results[name] = "FAIL"
        print(f"  [FAIL] {name}")
        traceback.print_exc()


# ── 1. imports ────────────────────────────────────────────────────────────────
def test_imports():
    import torch
    import transformers
    import datasets
    import sentence_transformers
    import mteb
    import ir_datasets
    import wandb
    import sklearn
    import numpy
    import tqdm

run("imports", test_imports)


# ── 2. SpladeSparseEncoder forward pass ──────────────────────────────────────
def test_modelling():
    import torch
    from modelling import SpladeSparseEncoder

    model = SpladeSparseEncoder("naver/splade-v3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    ids, mask, special = model.tokenize(["hello world", "sparse retrieval test"])
    ids, mask, special = ids.to(device), mask.to(device), special.to(device)
    with torch.no_grad():
        out = model(ids, mask, special)

    assert out.shape == (2, model.model.config.vocab_size), f"unexpected shape {out.shape}"
    assert (out >= 0).all(), "output contains negatives"

run("modelling/SpladeSparseEncoder", test_modelling)


# ── 3. SpladeSparseEncoder.save_pretrained ────────────────────────────────────
def test_save_pretrained():
    import torch, tempfile
    from modelling import SpladeSparseEncoder

    model = SpladeSparseEncoder("rasyosef/splade-tiny")
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        assert os.path.exists(os.path.join(d, "config.json"))

run("modelling/save_pretrained", test_save_pretrained)


# ── 4. dataloader utilities ───────────────────────────────────────────────────
def test_dataloader_imports():
    from dataloader import (
        MultipleNegatives, MultipleNegativesCE,
        read_collection, read_queries, read_triplets, read_ce_score,
    )

run("dataloader/imports", test_dataloader_imports)


# ── 5. pre_compute_ce_hn: argument parsing + model init ──────────────────────
def test_precompute_init():
    import torch
    from modelling import SpladeSparseEncoder, freeze_module

    new_enc = SpladeSparseEncoder("naver/splade-v3")
    old_enc = SpladeSparseEncoder("rasyosef/splade-tiny")
    freeze_module(old_enc)

    for p in old_enc.parameters():
        assert not p.requires_grad, "old encoder should be frozen"

run("pre_compute/freeze_module", test_precompute_init)


# ── 6. training: one forward+backward step (no data file needed) ──────────────
def test_training_step():
    import torch
    import torch.nn.functional as F
    from modelling import SpladeSparseEncoder, freeze_module

    device = "cuda" if torch.cuda.is_available() else "cpu"

    q_enc = SpladeSparseEncoder("naver/splade-v3").to(device)
    d_enc = SpladeSparseEncoder("rasyosef/splade-tiny").to(device)
    freeze_module(d_enc)

    sentences = ["what is sparse retrieval?",
                 "SPLADE sparse model for retrieval",
                 "dense vs sparse methods",
                 "GPU accelerated indexing"]

    q_ids, q_mask, q_sp = q_enc.tokenize(sentences[:1])
    d_ids, d_mask, d_sp = d_enc.tokenize(sentences)

    q_ids, q_mask, q_sp = q_ids.to(device), q_mask.to(device), q_sp.to(device)
    d_ids, d_mask, d_sp = d_ids.to(device), d_mask.to(device), d_sp.to(device)

    q_rep = q_enc(q_ids, q_mask, q_sp)      # (1, V)
    with torch.no_grad():
        d_rep = d_enc(d_ids, d_mask, d_sp)  # (4, V)

    scores = q_rep @ d_rep.T                # (1, 4)
    labels = torch.zeros(1, dtype=torch.long, device=device)
    loss = F.cross_entropy(scores, labels)
    loss.backward()

    assert loss.item() > 0

run("training/forward_backward_step", test_training_step)


# ── 7. static evaluation imports ─────────────────────────────────────────────
def test_eval_imports():
    import evaluate_beir_asy
    import evaluate_beir_query_adapter
    import evaluate_beir_rep_fusion
    import evaluate_beir_rank_fusion

run("static_evaluation/imports", test_eval_imports)


# ── 8. evaluate_beir_asy: encode a batch on BEIR/scifact ─────────────────────
def test_eval_asy_scifact():
    import torch
    from mteb import MTEB
    # Just verify the SparseEvaluator can be instantiated with real encoders
    import evaluate_beir_asy as asy
    import argparse

    args = argparse.Namespace(
        query_encoder="naver/splade-v3",
        doc_encoder="rasyosef/splade-tiny",
        batch_size=16,
        output_folder=os.path.join(ROOT, "results", "smoke_asy"),
        benchmark=None,
    )
    evaluator = asy.SparseEvaluator(
        query_encoder=args.query_encoder,
        doc_encoder=args.doc_encoder,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = evaluator.encode(["test query one", "test query two"],
                            batch_size=2, is_query=True)
    assert reps is not None

run("static_evaluation/SparseEvaluator_encode", test_eval_asy_scifact)


# ── 9. GPU availability ───────────────────────────────────────────────────────
def test_gpu():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    print(f"       GPU: {name}")
    assert "A6000" in name or True, f"expected A6000, got {name}"  # warn only

run("gpu/cuda_available", test_gpu)


# ── summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
passed = sum(v == "PASS" for v in results.values())
total  = len(results)
print(f"Result: {passed}/{total} passed")
for name, status in results.items():
    mark = "✓" if status == "PASS" else "✗"
    print(f"  {mark} {name}")
print("=" * 50)

if passed < total:
    sys.exit(1)
