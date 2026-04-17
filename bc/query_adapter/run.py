import os
import argparse
from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataloader import MultipleNegatives, read_collection, read_queries, read_triplets, read_qrels, read_ce_score, read_precompute_scores, MultipleNegativesCE
import wandb
import random
from modelling_splade import SpladeSparseEncoder, freeze_module


def zscore(x, eps=1e-6):
    m = x.mean(dim=1, keepdim=True)
    v = x.var(dim=1, keepdim=True, unbiased=False)
    return (x - m) / torch.sqrt(v + eps)

def kl_distill(student_scores, teacher_scores, T=1.0):
    logp_s = F.log_softmax(student_scores / T, dim=1)
    p_t = F.softmax(teacher_scores / T, dim=1).detach()
    return F.kl_div(logp_s, p_t, reduction="batchmean") * (T * T)

def margin_mse(student_scores, teacher_scores):
    s_pos = student_scores[:, :1]
    t_pos = teacher_scores[:, :1].detach()
    s_margin = s_pos - student_scores[:, 1:]
    t_margin = t_pos - teacher_scores[:, 1:].detach()
    return F.mse_loss(s_margin, t_margin)


@dataclass
class PrecomputeBatch:
    q_input_ids: torch.Tensor
    q_attention_mask: torch.Tensor
    q_special_tokens_mask: torch.Tensor
    d_input_ids: torch.Tensor
    d_attention_mask: torch.Tensor
    d_special_tokens_mask: torch.Tensor
    group_size: int 
    docs_scores: torch.Tensor

def collate_batch(
    items: List[tuple[str, List[str]]],
    tokenizer,
    max_q_len: int = 32,
    max_d_len: int = 256,
) -> PrecomputeBatch:
    queries = [item[0] for item in items]
    docs = [doc for item in items for doc in item[1]]
    docs_scores = [score for item in items for score in item[2]]
    group_size = len(items[0][1])

    q = tokenizer(
        queries, padding=True, truncation=True, max_length=max_q_len, return_tensors="pt", return_special_tokens_mask=True
    )

    d = tokenizer(
        docs, padding=True, truncation=True, max_length=max_d_len, return_tensors="pt", return_special_tokens_mask=True
    )

    return PrecomputeBatch(
        q_input_ids=q["input_ids"],
        q_attention_mask=q["attention_mask"],
        q_special_tokens_mask=q["special_tokens_mask"],
        d_input_ids=d["input_ids"],
        d_attention_mask=d["attention_mask"],
        d_special_tokens_mask=d["special_tokens_mask"],
        group_size=group_size,
        docs_scores=torch.tensor(docs_scores, dtype=torch.float32)
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
    tokenizer = AutoTokenizer.from_pretrained(doc_encoder_old_ckpt, use_fast=True)

    # Models
    doc_old = SpladeSparseEncoder(doc_encoder_old_ckpt).to(device)
    q_new = SpladeSparseEncoder(query_encoder_init_ckpt).to(device)
    freeze_module(doc_old)
    doc_old.eval()

    teacher_scores = read_precompute_scores(ce_path=triplet_path)
    docs = read_collection(collection_path)
    queries = read_queries(queries_path)
    qrels = read_qrels("msmarco-passage/train/judged")

    q_dict = {}
    query2pos_and_hardnegs_docids = {}
    query2pos_and_hardnegs_scores = {}

    for qid in teacher_scores:
        q_dict[qid] = queries[qid]

        # qrels[qid] can be a list/tuple/set of positive docids; pick the first one
        pos_candidates = qrels[qid]
        if isinstance(pos_candidates, (list, tuple, set)):
            pos_candidates = list(pos_candidates)
            if len(pos_candidates) == 0:
                continue
            pos_docid = pos_candidates[0]
        else:
            pos_docid = pos_candidates

        # Safely fetch and remove the positive score; default to 0.0 if missing
        pos_docid_score = teacher_scores[qid].pop(pos_docid, 0.0)

        # Remaining entries are hard negatives
        query2pos_and_hardnegs_docids[qid] = [pos_docid] + list(teacher_scores[qid].keys())
        query2pos_and_hardnegs_scores[qid] = [pos_docid_score] + list(teacher_scores[qid].values())
    

    # Split train/val
    query_ids = list(q_dict.keys())
    random.seed(42)
    random.shuffle(query_ids)

    val_ids = query_ids[:val_size] if val_size > 0 else []
    train_ids = query_ids[val_size:] if val_size > 0 else query_ids
    train_q_dict = {k: q_dict[k] for k in train_ids}
    
    train_query2pos_and_hardnegs_docids = {k: query2pos_and_hardnegs_docids[k] for k in train_ids}
    train_query2pos_and_hardnegs_scores = {k: query2pos_and_hardnegs_scores[k] for k in train_ids}
    val_q_dict = {k: q_dict[k] for k in val_ids}
    val_query2pos_and_hardnegs_docids = {k: query2pos_and_hardnegs_docids[k] for k in val_ids}
    val_query2pos_and_hardnegs_scores = {k: query2pos_and_hardnegs_scores[k] for k in val_ids}

    # Datasets & loaders
    train_ds = MultipleNegativesCE(
        docs=docs,
        q_dict=train_q_dict,
        query2hardnegs=train_query2pos_and_hardnegs_docids,
        query2hardnegs_scores=train_query2pos_and_hardnegs_scores,
        num_pos_and_negs= num_negs,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda items: collate_batch(items, tokenizer, max_q_len, max_d_len),
    )
    val_ds = MultipleNegativesCE(
        docs=docs,
        q_dict=val_q_dict,
        query2hardnegs=val_query2pos_and_hardnegs_docids,
        query2hardnegs_scores=val_query2pos_and_hardnegs_scores,
        num_pos_and_negs= num_negs,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda items: collate_batch(items, tokenizer, max_q_len, max_d_len),
    )

    optim = torch.optim.AdamW(q_new.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Initial validation
    if val_dl is not None:
        q_new.eval()
        val_loss, val_mrr, val_r1, steps = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Initial Validation"):
                batch = PrecomputeBatch(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.__dict__.items()})
                teacher_scores = batch.docs_scores.view(batch.q_input_ids.size(0), batch.group_size)
                    
                with torch.cuda.amp.autocast(enabled=use_amp):
                    d_vec_flat = doc_old(input_ids=batch.d_input_ids, attention_mask=batch.d_attention_mask,
                                         special_tokens_mask=batch.d_special_tokens_mask)

                    d_vec = d_vec_flat.view(batch.q_input_ids.size(0), batch.group_size, -1)

                    q_vec = q_new(input_ids=batch.q_input_ids, attention_mask=batch.q_attention_mask,
                                  special_tokens_mask=batch.q_special_tokens_mask)

                    fusion_scores = torch.einsum("bv,bgv->bg", q_vec, d_vec) / max(temperature, 1e-6)
                    # loss =  F.kl_div(F.log_softmax(fusion_scores, dim=1), F.softmax(teacher_scores, dim=1), reduction='batchmean') * (temperature ** 2)

                    kl_loss = F.kl_div(F.log_softmax(fusion_scores, dim=1), F.softmax(teacher_scores, dim=1), reduction='batchmean') * (temperature ** 2)
                    mse_loss = margin_mse(fusion_scores, teacher_scores)
                    loss = kl_loss + 0.02 * mse_loss

                    sorted_idx = fusion_scores.argsort(dim=1, descending=True)
                    pos_rank = (sorted_idx == 0).nonzero(as_tuple=True)[1] + 1
                    mrr = (1.0 / pos_rank.float()).mean().item()
                    r1 = (pos_rank == 1).float().mean().item()

                val_loss += loss.item()
                val_mrr += mrr
                val_r1 += r1
                steps += 1

        print(f"Init in-batch val_loss={val_loss/steps:.4f} val_mrr={val_mrr/steps:.4f} val_recall@1={val_r1/steps:.4f}")
        wandb.log({"eval/val_loss": val_loss/steps, "eval/val_mrr": val_mrr/steps, "eval/val_recall@1": val_r1/steps, "epoch": -1})

    # Training loop
    q_new.train()
    global_step = 0
    for ep in range(epochs):
        running_loss = 0.0
        running_mlm_loss = 0.0
        running_mlp_loss = 0.0
        running_kl_loss = 0.0
        running_mse_loss = 0.0
        dl_iter = iter(train_dl)

        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {ep+1}/{epochs}"):
            try:
                batch = next(dl_iter)
            except StopIteration:
                break

            global_step += 1
            batch = PrecomputeBatch(**{k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.__dict__.items()})

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    d_vec_flat = doc_old(input_ids=batch.d_input_ids, attention_mask=batch.d_attention_mask,
                                         special_tokens_mask=batch.d_special_tokens_mask)
                    teacher_scores = batch.docs_scores.view(batch.q_input_ids.size(0), batch.group_size)
                    
            with torch.cuda.amp.autocast(enabled=use_amp):
                q_vec_new = q_new(input_ids=batch.q_input_ids, attention_mask=batch.q_attention_mask,
                                special_tokens_mask=batch.q_special_tokens_mask)

                d_vec = d_vec_flat.view(batch.q_input_ids.size(0), batch.group_size, -1)

                fusion_scores = torch.einsum("bv,bgv->bg", q_vec_new, d_vec) / max(temperature, 1e-6)
                # loss = F.kl_div(F.log_softmax(fusion_scores, dim=1), F.softmax(teacher_scores, dim=1), reduction='batchmean') * (temperature ** 2)

                kl_loss = F.kl_div(F.log_softmax(fusion_scores, dim=1), F.softmax(teacher_scores, dim=1), reduction='batchmean') * (temperature ** 2)
                mse_loss = margin_mse(fusion_scores, teacher_scores)
                loss = kl_loss + 0.02 * mse_loss


            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(q_new.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()

            running_loss += loss.item()
            running_kl_loss += kl_loss.item()
            running_mse_loss += mse_loss.item()
            # running_mlm_loss += sparse_mlm_loss.item()
            # running_mlp_loss += sparse_mlp_loss.item()
            if global_step % 100 == 0:
                print(f"Step {global_step} | Train loss: {running_loss/100:.4f}")
                wandb.log({"train/train_loss": running_loss/100, 
                           "train/train_fusion_loss": running_loss/100,
                           "train/train_kl_loss": running_kl_loss/100,
                           "train/train_mse_loss": running_mse_loss/100,
                        #    "train/train_mlm_loss": running_mlm_loss/100,
                        #    "train/train_mlp_loss": running_mlp_loss/100,
                           "step": global_step}
                           )
                
                running_loss = 0.0
                running_mlm_loss = 0.0
                running_mlp_loss = 0.0
                running_kl_loss = 0.0
                running_mse_loss = 0.0

        # Validation after epoch
        if val_dl is not None:
            q_new.eval()
            val_loss, val_mrr, val_r1, steps = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for batch in tqdm(val_dl, desc=f"Validation Epoch {ep+1}"):
                    batch = PrecomputeBatch(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.__dict__.items()})

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        d_vec_flat = doc_old(input_ids=batch.d_input_ids, attention_mask=batch.d_attention_mask,
                                             special_tokens_mask=batch.d_special_tokens_mask)

                        q_vec = q_new(input_ids=batch.q_input_ids, attention_mask=batch.q_attention_mask,
                                        special_tokens_mask=batch.q_special_tokens_mask)
                        d_vec = d_vec_flat.view(batch.q_input_ids.size(0), batch.group_size, -1)
                        
                        scores = torch.einsum("bv,bgv->bg", q_vec, d_vec) / max(temperature, 1e-6)
                        # loss =  F.kl_div(F.log_softmax(scores, dim=1), F.softmax(batch.docs_scores.view(batch.q_input_ids.size(0), batch.group_size), dim=1), reduction='batchmean') * (temperature ** 2)

                        kl_loss = F.kl_div(F.log_softmax(scores, dim=1), F.softmax(teacher_scores, dim=1), reduction='batchmean') * (temperature ** 2)
                        mse_loss = margin_mse(scores, teacher_scores)
                        loss = kl_loss + 0.02 * mse_loss


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
    parser.add_argument("--triplet_path", default="/ivi/ilps/personal/jqiao/lsr-bc/data/train_set/splade_v3_precompute_ce_hn.jsonl.gz", type=str)
    parser.add_argument("--neg_source", default="bm25", type=str)
    parser.add_argument("--num_negs", default=21, type=int, help="More negatives = better, 7-15 common")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--lambda_sparsity", default=0, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--steps_per_epoch", default=5000, type=int)
    parser.add_argument("--val_size", default=5000, type=int)

    args = parser.parse_args()
    wandb.init(project="splade-query-distillation", config=vars(args))
    train_query_only(**vars(args))


if __name__ == "__main__":
    main()