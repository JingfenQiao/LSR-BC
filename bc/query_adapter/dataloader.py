import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import ir_datasets
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Optional
from datasets import load_dataset
from tqdm import tqdm
import gzip
import pickle
from collections import defaultdict

def read_hard_negatives(file_path):
    hard_negatives = defaultdict(dict)
    with open(file_path, "r") as f:
        for index, line in tqdm(enumerate(f), desc="Reading hard negatives"):
            # if index > 10:break
            parts = line.strip().split(" ")
            # print(parts)
            query_id = parts[0]
            doc = parts[2]
            score = float(parts[4])
            hard_negatives[query_id][doc] = score

    query2hardnegs: DefaultDict[str, List[str]] = defaultdict(list)
    query2hardnegs_scores: DefaultDict[str, List[float]] = defaultdict(list)

    for qid in tqdm(hard_negatives, desc=f"Preprocessing hard negatives from {file_path}"):
        qid_str = str(qid)
        negs_items = [(str(did), score) for did, score in hard_negatives[qid].items()]
        sorted_negs = sorted(negs_items, key=lambda x: x[1], reverse=True)
        query2hardnegs[qid_str] = [did for did, score in sorted_negs]
        query2hardnegs_scores[qid_str] = [score for did, score in sorted_negs]

    return query2hardnegs, query2hardnegs_scores   


def read_collection(collection_path: str, text_fields=["text"]):
    dataset = ir_datasets.load(collection_path)
    doc_dict = {}
    for doc in tqdm(
        dataset.docs_iter(),
        desc=f"Loading collection from ir_datasets: {collection_path}", 
    ):
        texts = [getattr(doc, field) for field in text_fields]
        text = " ".join(texts)
        doc_dict[doc.doc_id] = text
    return doc_dict


def read_qrels(qrels_path: str):
    qrels: DefaultDict[str, List[str]] = defaultdict(list)
    dataset = ir_datasets.load(qrels_path)
    for qrel in tqdm(
        dataset.qrels_iter(),
        desc=f"Loading qrels from ir_datasets: {qrels_path}",
    ):
        qid = qrel.query_id
        docid = qrel.doc_id
        if qrel.relevance > 0:
            qrels[qid].append(docid)
    return qrels

def read_queries(queries_path: str, text_fields=["text"]):
    queries = []
    dataset = ir_datasets.load(queries_path)
    for query in tqdm(
        dataset.queries_iter(),
        desc=f"Loading queries from ir_datasets: {queries_path}",
    ):
        query_id = query.query_id
        texts = [getattr(query, field) for field in text_fields]
        text = " ".join(texts) 
        queries.append((query_id, text))
    return queries

# def read_ce_score(ce_path: str):
#     query2hardnegs: DefaultDict[str, List[str]] = defaultdict(list)
#     query2hardnegs_scores: DefaultDict[str, List[str]] = defaultdict(list)
#     with gzip.open(ce_path, "rb") as f:
#         data = pickle.load(f)

#         for qid in tqdm(data, desc=f"Preprocessing data from {ce_path}"):
#             query2hardnegs[qid] = list(data[qid].keys())
#             query2hardnegs_scores[qid] = list(data[qid].values())
#     return query2hardnegs, query2hardnegs_scores


def read_ce_score(ce_path: str):
    # score_filename = "/ivi/ilps/personal/jkang1/jf/lsr-bc/data/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
    with gzip.open(ce_path, "rb") as f:
        data = pickle.load(f)

    query2hardnegs: DefaultDict[str, List[str]] = defaultdict(list)
    query2hardnegs_scores: DefaultDict[str, List[float]] = defaultdict(list)

    for qid in tqdm(data, desc=f"Preprocessing CE scores from {ce_path}"):
        qid_str = str(qid)
        negs_items = [(str(did), score) for did, score in data[qid].items()]
        sorted_negs = sorted(negs_items, key=lambda x: x[1], reverse=True)
        query2hardnegs[qid_str] = [did for did, score in sorted_negs]
        query2hardnegs_scores[qid_str] = [score for did, score in sorted_negs]

    return query2hardnegs, query2hardnegs_scores


def read_triplets(
    dataset_name: str,
    neg_source: str = "bm25",
    max_negs_per_q: Optional[int] = 30,
) -> Tuple[List[Tuple[str, str, str]], DefaultDict[str, List[str]], DefaultDict[str, List[str]]]:

    ds = load_dataset(dataset_name, data_files="msmarco-hard-negatives-bm25_1k.jsonl.gz", split="train")
    query2pos: DefaultDict[str, List[str]] = defaultdict(list)
    query2neg: DefaultDict[str, List[str]] = defaultdict(list)
    triplets: List[Tuple[str, str, str]] = []

    for index, row in tqdm(enumerate(ds), desc=f"Loading hard negatives ({neg_source})"):
        if index > 2000000: break
        if row is None: continue
        qid = str(row["qid"])
        pos_list = row.get("pos") or []
        pos = [str(p) for p in pos_list if p is not None]

        neg_field = row.get("neg") or {}
        if isinstance(neg_field, dict):
            negs_raw = neg_field.get(neg_source) or []
        else:
            negs_raw = neg_field

        negs = [str(n) for n in negs_raw if n is not None]
        if max_negs_per_q is not None and len(negs) > max_negs_per_q: negs = negs[:max_negs_per_q]

        query2pos[qid].extend(pos)
        query2neg[qid].extend(negs)

        for p in pos:
            for n in negs:
                triplets.append((qid, p, n))
    print(f"Total triplets loaded: {len(triplets)}, number of negatives per query capped at {max_negs_per_q}")
    return triplets, query2pos, query2neg


class MultipleNegatives(Dataset):
    def __init__(
        self,
        docs: Union[Dict[str, str]],
        q_dict: Dict[str, str],
        query2pos: Dict[str, List[str]],
        query2neg: Dict[str, List[str]],
        train_group_size: int,
    ):
        self.docs = docs
        self.is_dict = isinstance(self.docs, dict)
        self.query2pos = query2pos
        self.query2neg = query2neg
        self.q_dict = q_dict
        self.qids = list(self.q_dict.keys())
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.q_dict)

    def __getitem__(self, item):
        qid = self.qids[item]
        query = self.q_dict[qid]
        pos_id = random.choice(self.query2pos[qid])
        pos_psg = self.docs.get(pos_id)
        group_batch = [pos_psg]
        num_negs = self.train_group_size - 1

        if len(self.query2neg[qid]) < num_negs:
            neg_ids = random.choices(self.query2neg[qid], k=num_negs)
        else:
            neg_ids = random.sample(self.query2neg[qid], k=num_negs)
        neg_texts = []
        for neg_id in neg_ids:
            if self.is_dict:
                t = self.docs.get(neg_id)
            else:
                d = self.docs.get(neg_id)
                t = d.text if d else None
            if t is None:
                continue
            neg_texts.append(t)
        if len(neg_texts) < num_negs:
            # Could handle by sampling more, but for now assume enough
            raise ValueError("Not enough negs")
        group_batch.extend(neg_texts)
        return query, group_batch


class MultipleNegativesCE(Dataset):
    def __init__(
        self,
        docs: Union[Dict[str, str]],
        q_dict: Dict[str, str],
        query2hardnegs: Dict[str, List[str]],
        query2hardnegs_scores: Dict[str, List[float]],
        train_group_size: int,
    ):
        self.docs = docs
        self.is_dict = isinstance(self.docs, dict)
        self.query2hardnegs = query2hardnegs
        self.query2hardnegs_scores = query2hardnegs_scores
        self.q_dict = q_dict
        self.qids = list(self.q_dict.keys())
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.q_dict)

    def __getitem__(self, item):
        qid = self.qids[item]
        query = self.q_dict[qid]

        group_batch = []
        group_batch_scores = []
        num_negs = self.train_group_size - 1
        neg_ids = self.query2hardnegs[qid][:num_negs]
        group_batch_scores = self.query2hardnegs_scores[qid][:num_negs]

        neg_texts = []
        for neg_id in neg_ids:
            if self.is_dict:
                t = self.docs.get(neg_id)
            else:
                d = self.docs.get(neg_id)
                t = d.text if d else None
            if t is None:
                continue
            neg_texts.append(t)

        group_batch.extend(neg_texts)

        return query, group_batch, group_batch_scores

