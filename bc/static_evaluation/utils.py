import argparse
import torch
import numpy as np
import json
from transformers import AutoModel
from functools import partial
import mteb
from mteb.model_meta import ModelMeta
from typing import Union  # Add this import
import mteb
import json
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import logging
from typing import Dict, List, Iterable, Tuple
import ir_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Prefixes
IRDS_PREFIX = "irds:"

def read_collection(collection_path: str) -> Dict[str, str]:

    doc_dict: Dict[str, str] = {}
    if collection_path.startswith(IRDS_PREFIX):
        irds_name = collection_path[len(IRDS_PREFIX):]
        logger.info(f"Loading documents from ir_datasets: {irds_name}")
        dataset = ir_datasets.load(irds_name)
        for doc in tqdm(dataset.docs_iter(), desc="Loading docs"):
            doc_dict[doc.doc_id] = doc.text
    else:
        path = Path(collection_path)
        if not path.exists():
            raise FileNotFoundError(f"Collection not found: {collection_path}")
        logger.info(f"Reading local collection: {collection_path}")
        with path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading docs"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    logger.warning(f"Malformed line: {line}")
                    continue
                doc_dict[parts[0]] = parts[1]

    logger.info(f"Loaded {len(doc_dict):,} documents.")
    return doc_dict

def read_queries(queries_path: str) -> Dict[str, str]:
    

    if queries_path.startswith(IRDS_PREFIX):
        queries: Dict[str, str] = {}
        irds_name = queries_path[len(IRDS_PREFIX):]
        logger.info(f"Loading queries from ir_datasets: {irds_name}")
        dataset = ir_datasets.load(irds_name)
        for q in tqdm(dataset.queries_iter(), desc="Loading queries"):
            queries[q.query_id] = q.text
    else:
        path = Path(queries_path)
        if not path.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        logger.info(f"Reading local queries: {queries_path}")
        with path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading queries"):
                line = line.strip()
                if not line: continue
                parts = line.split("\t", 1)
                queries[str(parts[0])] = parts[1]

    logger.info(f"Loaded {len(queries):,} queries.")
    return queries

def read_qrels(qrels_path: str, text_fields: List[str] = None) -> Dict[str, Dict[str, int]]:
    if qrels_path.startswith(IRDS_PREFIX):
        qrels = defaultdict(dict)
        irds_name = qrels_path[len(IRDS_PREFIX):]
        logger.info(f"Loading qrels from ir_datasets: {irds_name}")
        dataset = ir_datasets.load(irds_name)
        for q in tqdm(dataset.qrels_iter(), desc="Loading queries"):
            qrels[str(q.query_id)][str(q.doc_id)] = int(q.relevance)
    else:
        path = Path(qrels_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        qrels = {
            str(qid): {str(docid): float(score) for docid, score in rels.items()} for qid, rels in data.items()
        }
    return qrels



def write_jsonl(items: Iterable[Dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def write_qrels_tsv(qrels: Dict[str, Dict[str, int]], path: Path):
    with path.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid in sorted(qrels.keys()):
            for docid, score in sorted(qrels[qid].items()):
                f.write(f"{qid}\t{docid}\t{score}\n")



def save_split(
    split_name: str,
    corpus: Dict[str, Dict],
    queries: Dict[str, Dict],
    qrels: Dict[str, Dict[str, int]],
    split_dir: Path,
    ratio_pct: int,
):
    prefix = f"{split_name}_{ratio_pct}"

    write_jsonl(
        [{"_id": did, "title": "", "text": d["text"]} for did, d in corpus.items()],
        split_dir / f"corpus_{prefix}.jsonl"
    )
    write_jsonl(
        [{"_id": qid, "text": q["text"]} for qid, q in queries.items()],
        split_dir / f"queries_{prefix}.jsonl"
    )
    write_qrels_tsv(qrels, split_dir / f"qrels_{prefix}.tsv")

    print(f"Saved '{split_name}' split: {len(corpus):,} docs, {len(queries):,} queries")
