import os
import argparse
from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from modelling import SpladeSparseEncoder, freeze_module  # assuming this file defines the class correctly
from dataloader import MultipleNegatives, read_collection, read_queries, read_triplets, read_qrels, read_ce_score
import wandb
import random

@dataclass
class Batch:
    q_input_ids: torch.Tensor
    q_attention_mask: torch.Tensor
    q_special_tokens_mask: torch.Tensor
    d_input_ids: torch.Tensor
    d_attention_mask: torch.Tensor
    d_special_tokens_mask: torch.Tensor
    group_size: int  # 1 + num_negs


def collate_batch(
    items: List[tuple[str, List[str]]],
    tokenizer,
    max_q_len: int = 32,
    max_d_len: int = 256,
) -> Batch:
    queries = [item[0] for item in items]
    docs = [doc for item in items for doc in item[1]]
    group_size = len(items[0][1])

    q = tokenizer(
        queries, padding=True, truncation=True, max_length=max_q_len, return_tensors="pt", return_special_tokens_mask=True
    )
    d = tokenizer(
        docs, padding=True, truncation=True, max_length=max_d_len, return_tensors="pt", return_special_tokens_mask=True
    )

    return Batch(
        q_input_ids=q["input_ids"],
        q_attention_mask=q["attention_mask"],
        q_special_tokens_mask=q["special_tokens_mask"],
        d_input_ids=d["input_ids"],
        d_attention_mask=d["attention_mask"],
        d_special_tokens_mask=d["special_tokens_mask"],
        group_size=group_size,
    )

def group_infonce(scores_bg: torch.Tensor) -> torch.Tensor:
    """
    scores_bg: (B, G) where doc group per query is [pos, neg1..negN]
    Positive index is 0.
    """
    targets = torch.zeros(scores_bg.size(0), dtype=torch.long, device=scores_bg.device)
    return F.cross_entropy(scores_bg, targets)


def get_nonzero_terms(
    x_bv: torch.Tensor,
    top_k: int = 20,
    min_value: float = 0.0,
):
    """
    x_bv: (V,) SPLADE vector
    Returns list of (vocab_id, value) sorted by value desc
    """
    nz = (x_bv > min_value).nonzero(as_tuple=True)[0]
    if nz.numel() == 0:
        return []

    values = x_bv[nz]
    topk = min(top_k, values.numel())
    vals, idx = torch.topk(values, k=topk)

    vocab_ids = nz[idx]
    return list(zip(vocab_ids.tolist(), vals.tolist()))


def train_query_only(
    *,
    doc_encoder_old_ckpt: str,
    query_encoder_init_ckpt: str,
    out_dir: str,
    collection_path: str = "msmarco-passage/train",
    queries_path: str = "msmarco-passage/train",
    triplet_path: str = "sentence-transformers/msmarco-hard-negatives",
    neg_source: str = "bm25",
    num_negs: int = 4,
    batch_size: int = 16,
    max_q_len: int = 32,
    max_d_len: int = 256,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    temperature: float = 1.0,
    lambda_sparsity: float = 1e-4,
    epochs: int = 1,
    steps_per_epoch: int = 10_000,
    num_workers: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    val_size: int = 1000,
):
    os.makedirs(out_dir, exist_ok=True)

    # Use SAME tokenizer from document encoder (reference side)
    tokenizer = AutoTokenizer.from_pretrained(doc_encoder_old_ckpt, use_fast=True)

    # Models
    doc_old = SpladeSparseEncoder(doc_encoder_old_ckpt).to(device)
    q_new = SpladeSparseEncoder(query_encoder_init_ckpt).to(device)
    teacher_model = SpladeSparseEncoder(query_encoder_init_ckpt).to(device)

    # Freeze document encoder
    freeze_module(doc_old)
    doc_old.eval()
    freeze_module(teacher_model)
    teacher_model.eval()

    docs = read_collection(collection_path)
    q_list = read_queries(queries_path)

    # Load data
    qrels = read_qrels("msmarco-passage/train")
    query2neg, _ = read_ce_score(ce_path=triplet_path)

    # Filter and clean negatives
    query2pos = {}
    for qid, pos_docs in qrels.items():
        neg_docs = query2neg.get(qid, [])
        
        # Skip if no negatives available
        if not neg_docs:
            continue
        
        # Remove positives from negatives (avoid overlap)
        pos_set = set(pos_docs)
        neg_docs = [docid for docid in neg_docs if docid not in pos_set]
        
        if not neg_docs:
            continue
        # Only keep this query if it has both pos and neg
        query2pos[qid] = pos_docs
        query2neg[qid] = neg_docs

    q_dict = {qid: text for qid, text in q_list if qid in query2pos}

    # Split train/val
    query_ids = list(q_dict.keys())
    random.seed(42)
    random.shuffle(query_ids)
    val_ids = query_ids[:val_size] if val_size > 0 else []
    train_ids = query_ids[val_size:] if val_size > 0 else query_ids

    if len(train_ids) == 0:
        print("No training data available.")
        return

    train_query2pos = {k: query2pos[k] for k in train_ids}
    train_query2neg = {k: query2neg[k] for k in train_ids}
    train_q_dict = {k: q_dict[k] for k in train_ids}

    # Datasets & loaders
    train_ds = MultipleNegatives(
        docs=docs,
        q_dict=train_q_dict,
        query2pos=train_query2pos,
        query2neg=train_query2neg,
        train_group_size=1 + num_negs,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda items: collate_batch(items, tokenizer, max_q_len, max_d_len),
    )

    val_dl = None
    if val_ids:
        val_query2pos = {k: query2pos[k] for k in val_ids}
        val_query2neg = {k: query2neg[k] for k in val_ids}
        val_q_dict = {k: q_dict[k] for k in val_ids}

        val_ds = MultipleNegatives(
            docs=docs,
            q_dict=val_q_dict,
            query2pos=val_query2pos,
            query2neg=val_query2neg,
            train_group_size=1 + num_negs,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda items: collate_batch(items, tokenizer, max_q_len, max_d_len),
        )

    # Optimizer & scheduler
    optim = torch.optim.AdamW(q_new.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # === SAVE INITIAL MODEL (before any training) ===
    init_dir = os.path.join(out_dir, "epoch-0")
    os.makedirs(init_dir, exist_ok=True)
    q_new.save_pretrained(init_dir)
    tokenizer.save_pretrained(init_dir)
    print(f"Saved initial query encoder: {init_dir}")

    # Initial validation
    if val_dl is not None:
        q_new.eval()
        val_loss, val_mrr, val_r1, steps = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Initial Validation"):
                batch = Batch(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.__dict__.items()})

                with torch.cuda.amp.autocast(enabled=use_amp):
                    d_vec_flat = doc_old(input_ids=batch.d_input_ids, attention_mask=batch.d_attention_mask,
                                         special_tokens_mask=batch.d_special_tokens_mask)
                    q_vec = q_new(input_ids=batch.q_input_ids, attention_mask=batch.q_attention_mask,
                                  special_tokens_mask=batch.q_special_tokens_mask)
                    d_vec = d_vec_flat.view(batch.q_input_ids.size(0), batch.group_size, -1)
                    
                    scores = torch.einsum("bv,bgv->bg", q_vec, d_vec) / max(temperature, 1e-6)
                    loss = group_infonce(scores)

                    sorted_idx = scores.argsort(dim=1, descending=True)
                    pos_rank = (sorted_idx == 0).nonzero(as_tuple=True)[1] + 1
                    mrr = (1.0 / pos_rank.float()).mean().item()
                    r1 = (pos_rank == 1).float().mean().item()

                val_loss += loss.item()
                val_mrr += mrr
                val_r1 += r1
                steps += 1

        print(f"Init in-batch val_loss={val_loss/steps:.4f} val_mrr={val_mrr/steps:.4f} val_recall@1={val_r1/steps:.4f}")
        wandb.log({"val_loss": val_loss/steps, "val_mrr": val_mrr/steps, "val_recall@1": val_r1/steps, "epoch": -1})

    # Training loop
    q_new.train()
    global_step = 0
    for ep in range(epochs):
        running_loss = 0.0
        dl_iter = iter(train_dl)

        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {ep+1}/{epochs}"):
            try:
                batch = next(dl_iter)
            except StopIteration:
                break

            global_step += 1
            batch = Batch(**{k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.__dict__.items()})

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    d_vec_flat = doc_old(input_ids=batch.d_input_ids, attention_mask=batch.d_attention_mask,
                                         special_tokens_mask=batch.d_special_tokens_mask)
                    
                    d_vec_flat_teacher = teacher_model(input_ids=batch.d_input_ids, attention_mask=batch.d_attention_mask,
                                         special_tokens_mask=batch.d_special_tokens_mask)
                    
                    q_vec_new_teacher = teacher_model(input_ids=batch.q_input_ids, attention_mask=batch.q_attention_mask,
                                    special_tokens_mask=batch.q_special_tokens_mask)
                    

            with torch.cuda.amp.autocast(enabled=use_amp):
                q_vec_new = q_new(input_ids=batch.q_input_ids, attention_mask=batch.q_attention_mask,
                              special_tokens_mask=batch.q_special_tokens_mask)

                d_vec = d_vec_flat.view(batch.q_input_ids.size(0), batch.group_size, -1)
                scores = torch.einsum("bv,bgv->bg", q_vec_new, d_vec) / max(temperature, 1e-6)

                d_vec_teacher = d_vec_flat_teacher.view(batch.q_input_ids.size(0), batch.group_size, -1)
                teacher_scores = torch.einsum("bv,bgv->bg", q_vec_new_teacher, d_vec_teacher) / max(temperature, 1e-6)
                loss = F.kl_div(F.log_softmax(scores, dim=1), F.softmax(teacher_scores, dim=1), reduction='batchmean') * (temperature ** 2)

                # loss = group_infonce(scores)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(q_new.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()

            running_loss += loss.item()
            if global_step % 100 == 0:
                print(f"Step {global_step} | Train loss: {running_loss/100:.4f}")
                wandb.log({"train_loss": running_loss/100, "step": global_step})
                running_loss = 0.0

        # Validation after epoch
        if val_dl is not None:
            q_new.eval()
            val_loss, val_mrr, val_r1, steps = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for batch in tqdm(val_dl, desc=f"Validation Epoch {ep+1}"):
                    batch = Batch(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.__dict__.items()})

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        d_vec_flat = doc_old(input_ids=batch.d_input_ids, attention_mask=batch.d_attention_mask,
                                             special_tokens_mask=batch.d_special_tokens_mask)
                        q_vec = q_new(input_ids=batch.q_input_ids, attention_mask=batch.q_attention_mask,
                                      special_tokens_mask=batch.q_special_tokens_mask)
                        d_vec = d_vec_flat.view(batch.q_input_ids.size(0), batch.group_size, -1)
                        scores = torch.einsum("bv,bgv->bg", q_vec, d_vec) / max(temperature, 1e-6)
                        loss = group_infonce(scores)

                        sorted_idx = scores.argsort(dim=1, descending=True)
                        pos_rank = (sorted_idx == 0).nonzero(as_tuple=True)[1] + 1
                        mrr = (1.0 / pos_rank.float()).mean().item()
                        r1 = (pos_rank == 1).float().mean().item()

                    val_loss += loss.item()
                    val_mrr += mrr
                    val_r1 += r1
                    steps += 1

            print(f"Epoch {ep+1} | in-batch val_loss={val_loss/steps:.4f} val_mrr={val_mrr/steps:.4f} val_recall@1={val_r1/steps:.4f}")
            wandb.log({"val_loss": val_loss/steps, "val_mrr": val_mrr/steps, "val_recall@1": val_r1/steps, "epoch": ep+1, "step": global_step})
            q_new.train()

        ckpt_dir = os.path.join(out_dir, f"epoch-{ep+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        q_new.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Saved checkpoint: {ckpt_dir}")
    print(f"Training complete. Final model saved in: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Distill SPLADE query encoder to frozen document encoder")
    parser.add_argument("--doc_encoder_old_ckpt", default="rasyosef/splade-tiny", type=str,
                        help="Frozen document encoder (defines target sparse space)")
    parser.add_argument("--query_encoder_init_ckpt", default="naver/splade-v3", type=str,
                        help="Initial weights for query encoder (will be tuned)")
    parser.add_argument("--out_dir", default="./new_model_as_teacher", type=str)
    # parser.add_argument("--triplet_path", default="sentence-transformers/msmarco-hard-negatives", type=str)
    parser.add_argument("--triplet_path", default="/ivi/ilps/personal/jkang1/jf/lsr-bc/data/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz", type=str)
    parser.add_argument("--neg_source", default="bm25", type=str)
    parser.add_argument("--num_negs", default=20, type=int, help="More negatives = better, 7-15 common")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--lambda_sparsity", default=0, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--steps_per_epoch", default=5000, type=int)
    parser.add_argument("--val_size", default=5000, type=int)

    args = parser.parse_args()
    wandb.init(project="splade-query-distillation", config=vars(args))
    # wandb.run.name = "ce_new_model_as_teacher"
    train_query_only(**vars(args))


if __name__ == "__main__":
    main()