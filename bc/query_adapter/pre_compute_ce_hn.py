import os
import argparse
from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from dataloader import MultipleNegatives, read_collection, read_queries, read_triplets
import random
from collections import defaultdict
from torch.utils.data import Dataset
from sentence_transformers import SparseEncoder
import gzip
import json
from collections import defaultdict
from dataloader import read_hard_negatives, read_qrels, read_collection, read_queries, read_triplets, read_ce_score

class MultipleNegatives(Dataset):
    def __init__(
        self,
        docs: Dict[str, str],
        q_dict: Dict[str, str],
        query2pos: Dict[str, List[str]],
        query2neg: Dict[str, List[str]],
        train_group_size: int,
    ):
        self.docs = docs
        self.q_dict = q_dict
        self.query2pos = query2pos
        self.query2neg = query2neg
        self.qids = list(self.q_dict.keys())
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.q_dict)

    def __getitem__(self, item):
        qid = self.qids[item]
        query = self.q_dict[qid]
        pos_id = random.choice(self.query2pos[qid])
        pos_psg = self.docs.get(pos_id)
        num_negs = self.train_group_size - 1

        if len(self.query2neg[qid]) < num_negs:
            neg_ids = random.choices(self.query2neg[qid], k=num_negs)
        else:
            neg_ids = random.sample(self.query2neg[qid], k=num_negs)
        neg_texts = [self.docs.get(neg_id) for neg_id in neg_ids]
        if len(neg_texts) < num_negs or any(t is None for t in neg_texts):
            raise ValueError("Not enough negs or missing texts")
        group_batch = [pos_psg] + neg_texts
        group_ids = [pos_id] + neg_ids
        return query, group_batch, group_ids, qid

@dataclass
class PrecomputeBatch:
    queries: List[str]
    docs: List[str]
    q_ids: List[str]
    d_ids: List[str]
    group_size: int

def collate_batch(
    items: List[tuple[str, List[str], List[str], str]],
    max_q_len: int = 32,
    max_d_len: int = 256,
) -> PrecomputeBatch:
    queries = [item[0] for item in items]
    docs = [doc for item in items for doc in item[1]]
    d_ids = [did for item in items for did in item[2]]
    q_ids = [item[3] for item in items]
    group_size = len(items[0][1])

    return PrecomputeBatch(
        queries=queries,
        docs=docs,
        q_ids=q_ids,
        d_ids=d_ids,
        group_size=group_size,
    )

def precompute(
    *,
    old_model_name: str,
    new_model_name: str,
    out_dir: str,
    collection_path: str = "msmarco-passage/train",
    queries_path: str = "msmarco-passage/train",
    triplet_path: str = "sentence-transformers/msmarco-hard-negatives",
    neg_source: str = "bm25",
    num_negs: int = 4,
    batch_size: int = 16,
    max_q_len: int = 32,
    max_d_len: int = 256,
    num_workers: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    new = SparseEncoder(new_model_name).to(device)
    new.eval()

    query2neg, _ = read_ce_score(ce_path=triplet_path, num_hard_negs=num_negs)
    docs = read_collection(collection_path)
    q_list = read_queries(queries_path)
    qrels = read_qrels("msmarco-passage/train/judged")
    
    # Filter and clean negatives
    query2pos = {}
    for qid, pos_docs in qrels.items():
        neg_docs = query2neg.get(qid, [])
        
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

    train_q_dict = {k: q_dict[k] for k in query_ids}

    # Datasets & loaders
    train_ds = MultipleNegatives(
        docs=docs,
        q_dict=train_q_dict,
        query2pos=query2pos,
        query2neg=query2neg,
        train_group_size=1 + num_negs,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda items: collate_batch(items, max_q_len, max_d_len),
    )

    with gzip.open(out_dir, 'wt') as out_f:
        with torch.no_grad():
            for batch in tqdm(train_dl, desc="Saving shards"):
                q_ids = batch.q_ids
                d_ids = batch.d_ids
                q_new = new.encode(batch.queries).to_dense().to(device)  # (B, V)
                d_new = new.encode(batch.docs).to_dense().to(device)  # (B*G, V)
                # d_old = old.encode(batch.docs).to_dense().to(device)  # (B*G, V)
                scores = torch.einsum("bv,bgv->bg", q_new, d_new.view(len(q_ids), batch.group_size, -1))  # (B, G)

                B = len(q_ids)
                G = batch.group_size
                for b in range(B):
                    for g in range(G):
                        idx = b * G + g
                        d_id = d_ids[idx]
                        sample = {
                            'q_id': q_ids[b],
                            'd_id': d_id,
                            # 'q_new_vec': q_new[b].cpu().tolist(),
                            # 'd_new_vec': d_new[idx].cpu().tolist(),
                            'score': scores[b, g].cpu().item()
                        }
                        out_f.write(json.dumps(sample) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Precompute data for SPLADE distillation")
    parser.add_argument("--doc_encoder_old_ckpt", default="rasyosef/splade-tiny", type=str,
                        help="Frozen document encoder (defines target sparse space)")
    parser.add_argument("--query_encoder_init_ckpt", default="naver/splade-v3", type=str,
                        help="Initial weights for query encoder")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--triplet_path", type=str)
    parser.add_argument("--neg_source", default="bm25", type=str)
    parser.add_argument("--num_negs", default=50, type=int, help="More negatives = better, 7-15 common")
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()
    precompute(
        old_model_name=args.doc_encoder_old_ckpt,
        new_model_name=args.query_encoder_init_ckpt,
        out_dir=args.out_dir,
        triplet_path=args.triplet_path,
        neg_source=args.neg_source,
        num_negs=args.num_negs,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()