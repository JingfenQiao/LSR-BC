import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse
import json
import orjson
from utils import read_collection, read_queries, read_qrels
from config_manager import BaselineConfigManager
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.search_engines import semantic_search_seismic
import ir_measures
from ir_measures import nDCG, RR, R, Success
import torch
import gc
import os
from tqdm import tqdm
from seismic import SeismicIndex
import ir_datasets
from collections import defaultdict
import pandas as pd
import gzip
from scipy import stats
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Note: Using ephemeral positional mapping (in-memory only, no persistent file)
# SeismicIndex uses positional indices 0,1,2,... and we map back using cumulative_doc_ids list

def convert_results_to_string_ids(results: List[List[Dict]], cumulative_doc_ids: List[str]) -> List[List[Dict]]:
    """Convert positional corpus IDs in results back to string IDs using cumulative list.
    
    Args:
        results: Search results with positional corpus_id (0, 1, 2, ...)
        cumulative_doc_ids: List mapping positions to actual document IDs
    """
    converted_results = []
    for query_results in results:
        converted_query_results = []
        for entry in query_results:
            positional_id = entry['corpus_id']
            string_id = cumulative_doc_ids[positional_id]
            converted_query_results.append({
                'corpus_id': string_id,
                'score': entry['score']
            })
        converted_results.append(converted_query_results)
    return converted_results

def encode_and_save_to_jsonl(
    model: SparseEncoder,
    texts: List[str],
    ids: List[str],
    output_jsonl: str,
    batch_size: int = 32,
    chunk_size: int = 10000,
    is_query: bool = False,
    description: str = "Encoding",
):
    """
    Encode texts and save directly to JSONL file.
    Works for both queries and documents.
    For sparse models, only non-zero weights are stored.
    """
    encode_func = model.encode_query if is_query else model.encode_document
    
    with open(output_jsonl, 'w') as f:
        num_chunks = (len(texts) - 1) // chunk_size + 1
        for i in tqdm(range(num_chunks), desc=description):
            start = i * chunk_size
            end = min(start + chunk_size, len(texts))
            chunk_texts = texts[start:end]
            chunk_ids = ids[start:end]
            
            with torch.no_grad():
                chunk_embs = encode_func(
                    chunk_texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_sparse_tensor=True,
                )
            chunk_decoded = model.decode(chunk_embs)
            
            for (item_id, emb) in zip(chunk_ids, chunk_decoded):
                # For sparse embeddings, only store non-zero values
                # emb is already a list of (token, weight) tuples from decode()
                # Filter out zero or near-zero weights to save space
                sparse_vector = {token: float(weight) for token, weight in emb if abs(weight) > 1e-7}
                
                item_data = {
                    "id": str(item_id),
                    "vector": sparse_vector  # This contains only non-zero terms
                }
                json.dump(item_data, f)
                f.write('\n')
            
            del chunk_embs, chunk_decoded
            torch.cuda.empty_cache()
            gc.collect()

# def create_mixed_jsonl(old_jsonl: str, new_jsonl: str, mixed_jsonl: str, split_idx: int):
#     """
#     Create mixed JSONL by taking first split_idx lines from old_jsonl and the rest from new_jsonl.
#     """
#     with open(mixed_jsonl, 'w') as file, open(old_jsonl, 'r') as old_model, open(new_jsonl, 'r') as new_model:
#         for _ in range(split_idx):
#             file.write(old_model.readline())
#         for _ in range(split_idx):
#             new_model.readline()  # Skip first split_idx in new
#         for line in new_model:
#             file.write(line)

def rank_fusion_minmax(
    scores_1: Dict[str, float], 
    scores_2: Dict[str, float], 
    k: int = 100,
    weight_1: float = 0.5,
    weight_2: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Min-Max normalization fusion to combine two score distributions.
    
    Args:
        scores_1: Dictionary mapping doc_id -> score from first model
        scores_2: Dictionary mapping doc_id -> score from second model
        k: Number of results to return
        weight_1: Weight for first model (default 0.5)
        weight_2: Weight for second model (default 0.5)
    
    Returns:
        List of (doc_id, fused_score) tuples in fused rank order
    """
    # Get all unique docs
    all_docs = set(scores_1.keys()) | set(scores_2.keys())
    
    # Compute min/max for normalization
    if scores_1:
        min_1 = min(scores_1.values())
        max_1 = max(scores_1.values())
        range_1 = max(max_1 - min_1, 1e-9)  # Avoid division by zero
    else:
        min_1, max_1, range_1 = 0, 0, 1
    
    if scores_2:
        min_2 = min(scores_2.values())
        max_2 = max(scores_2.values())
        range_2 = max(max_2 - min_2, 1e-9)
    else:
        min_2, max_2, range_2 = 0, 0, 1
    
    # Compute fused scores with min-max normalization
    fused_scores = {}
    for doc in all_docs:
        score = 0.0
        
        if doc in scores_1:
            normalized_1 = (scores_1[doc] - min_1) / range_1
            score += weight_1 * normalized_1
        
        if doc in scores_2:
            normalized_2 = (scores_2[doc] - min_2) / range_2
            score += weight_2 * normalized_2
        
        fused_scores[doc] = score
    
    # Sort by fused score descending and return top-k
    fused_ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return fused_ranked

def representation_fusion(
    old_reps_decoded: List[List[Tuple[str, float]]], 
    new_reps_decoded: List[List[Tuple[str, float]]],
    weight_old: float = 0.5,
    weight_new: float = 0.5
) -> List[List[Tuple[str, float]]]:
    """
    Fuse two sets of sparse representations by averaging token weights.
    
    Args:
        old_reps_decoded: List of sparse vectors from old model, each as list of (token, weight) tuples
        new_reps_decoded: List of sparse vectors from new model, each as list of (token, weight) tuples
        weight_old: Weight for old model representations (default 0.5)
        weight_new: Weight for new model representations (default 0.5)
    
    Returns:
        List of fused sparse vectors, each as list of (token, weight) tuples
    """
    if len(old_reps_decoded) != len(new_reps_decoded):
        raise ValueError(f"Mismatch in number of representations: {len(old_reps_decoded)} vs {len(new_reps_decoded)}")
    
    all_reps = []
    
    for old_sparse, new_sparse in zip(old_reps_decoded, new_reps_decoded):
        # Convert to dictionaries for easier manipulation
        old_dict = dict(old_sparse)
        new_dict = dict(new_sparse)
        
        # Get union of all tokens
        union_tokens = set(old_dict.keys()) | set(new_dict.keys())
        
        # Compute weighted average for each token
        fused_reps = []
        for token in sorted(union_tokens):
            avg_weight = (old_dict.get(token, 0) * weight_old + new_dict.get(token, 0) * weight_new)
            if avg_weight > 0:  # Only keep non-zero weights
                fused_reps.append((token, avg_weight))
        
        all_reps.append(fused_reps)
    
    return all_reps
      
def perform_search(
    query_embeddings_decoded: List,
    query_embeddings_decoded_old: List,
    corpus_embeddings_decoded: List = None,
    top_k: int = 100,
    corpus_index = None, 
    use_rank_fusion: bool = False,
    save_rank_fusion_splits: bool = False,
    rank_fusion_output_dir: Path = None,
) -> Tuple[List[List[Dict]], float, Dict]:
    """
    Perform semantic search using Seismic with a single index.
    Returns results, search_time, additional_info.
    """
    
    # If fusion not needed, return new results only
    if not use_rank_fusion or query_embeddings_decoded_old is None:
        results, search_time = semantic_search_seismic(
            query_embeddings_decoded=query_embeddings_decoded,
            corpus_index=corpus_index,
            corpus_embeddings_decoded=corpus_embeddings_decoded,
            top_k=top_k,
            output_index=False,
        )
        return results, search_time
    
    # For rank fusion: Combine both query sets BEFORE searching
    # This ensures we only build/use the index once
    combined_query_embs = query_embeddings_decoded + query_embeddings_decoded_old
    
    # Single search call with combined queries
    combined_results, search_time = semantic_search_seismic(
        query_embeddings_decoded=combined_query_embs,
        corpus_index=corpus_index,
        corpus_embeddings_decoded=corpus_embeddings_decoded,
        top_k=top_k,
        output_index=False,
    )
    
    # Split results back into new and old
    num_new_queries = len(query_embeddings_decoded)
    results_new = combined_results[:num_new_queries]
    results_old = combined_results[num_new_queries:]

    # Optionally persist split results for inspection/debugging.
    if save_rank_fusion_splits and rank_fusion_output_dir is not None:
        output_dir = Path(rank_fusion_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        new_file = output_dir / "rank_fusion_results_new.jsonl.gz"
        old_file = output_dir / "rank_fusion_results_old.jsonl.gz"

        with gzip.open(new_file, "wt") as f_new:
            for query_idx, query_results in enumerate(results_new):
                row = {"query_idx": query_idx, "results": query_results}
                f_new.write(orjson.dumps(row).decode("utf-8") + "\n")

        with gzip.open(old_file, "wt") as f_old:
            for query_idx, query_results in enumerate(results_old):
                row = {"query_idx": query_idx, "results": query_results}
                f_old.write(orjson.dumps(row).decode("utf-8") + "\n")

        print(f"  Saved pre-fusion NEW results to {new_file}")
        print(f"  Saved pre-fusion OLD results to {old_file}")
    
    
    # Apply rank fusion for each query pair
    fused_results = []
    for new_res, old_res in zip(results_new, results_old):
        # Extract scores from both rankings
        new_scores = {entry['corpus_id']: entry['score'] for entry in new_res}
        old_scores = {entry['corpus_id']: entry['score'] for entry in old_res}
        
        # Apply min-max normalization fusion
        fused_ranked = rank_fusion_minmax(old_scores, new_scores, k=top_k)
        
        # Convert to result format
        fused_entries = [
            {'corpus_id': doc_id, 'score': score}
            for doc_id, score in fused_ranked
        ]
        
        fused_results.append(fused_entries)
    return fused_results, search_time

def perform_search_multiple_indexes(
    query_embeddings_decoded: List,
    corpus_indexes: List,  # List of SeismicIndex objects, one per session
    top_k: int = 100,
    merge_method: str = 'max',
) -> Tuple[List[List[Dict]], float]:
    """
    Perform semantic search using multiple Seismic indexes (one per session).
    Similar to the approach in share_with_gabrielle/search_w_multiple.py.
    
    Args:
        query_embeddings_decoded: Query embeddings
        corpus_indexes: List of SeismicIndex objects (one per session)
        top_k: Number of top results to retrieve per index
        merge_method: Method to merge results ('max' or 'rrf')
    
    Returns:
        Merged results, total search time
    """
    import time
    
    total_search_time = 0
    all_raw_scores = []  # List of results from each index
    
    # Search each index independently
    for idx_num, index in enumerate(corpus_indexes):
        start_time = time.time()
        results, search_time = semantic_search_seismic(
            query_embeddings_decoded=query_embeddings_decoded,
            corpus_index=index,
            top_k=top_k,
            output_index=False,
        )
        total_search_time += search_time
        all_raw_scores.append(results)
        print(f"    Searched index {idx_num+1}/{len(corpus_indexes)}: {len(results)} queries, {search_time:.2f}s")
    
    # Merge results from all indexes
    print(f"  Merging results from {len(corpus_indexes)} indexes using '{merge_method}' method...")
    merged_results = []
    
    for query_idx in range(len(query_embeddings_decoded)):
        if merge_method == 'max':
            # Use 'max' merge: later scores overwrite earlier ones (like dict.update())
            merged_scores = {}
            for index_results in all_raw_scores:
                for entry in index_results[query_idx]:
                    doc_id = entry['corpus_id']
                    score = entry['score']
                    # Update overwrites - if doc appears in multiple indexes, last score wins
                    merged_scores[doc_id] = score
            
            # Sort by score and take top_k
            sorted_results = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            merged_query_results = [
                {'corpus_id': doc_id, 'score': score}
                for doc_id, score in sorted_results
            ]
        
        elif merge_method == 'rrf':
            # Reciprocal Rank Fusion across indexes
            all_doc_ids = set()
            for index_results in all_raw_scores:
                for entry in index_results[query_idx]:
                    all_doc_ids.add(entry['corpus_id'])
            
            # Build rankings from each index
            rankings = []
            for index_results in all_raw_scores:
                ranking = [entry['corpus_id'] for entry in index_results[query_idx]]
                rankings.append(ranking)
            
            # Apply RRF with constant=60
            rrf_scores = {doc_id: 0 for doc_id in all_doc_ids}
            for ranking in rankings:
                for rank, doc_id in enumerate(ranking, 1):
                    rrf_scores[doc_id] += 1 / (rank + 60)
            
            # Sort by RRF score
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            merged_query_results = [
                {'corpus_id': doc_id, 'score': score}
                for doc_id, score in sorted_results
            ]
        
        else:
            raise ValueError(f"Unsupported merge_method: {merge_method}")
        
        merged_results.append(merged_query_results)
    
    return merged_results, total_search_time

def write_trec_results(
    filepath: Path,
    results: List[List[Dict]],
    query_ids: List[str],
):
    """
    Write search results in TREC format.
    """
    with open(filepath, "w") as f:
        for qid, result in zip(query_ids, results):
            rank = 1
            for entry in result:
                did = entry['corpus_id']
                score = entry['score']
                f.write(f"{str(qid)} Q0 {str(did)} {rank} {score} standard\n")
                rank += 1
    print(f"\nTREC results written to {filepath}")

def convert_results_to_run(
    results: List[List[Dict]],
    query_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Convert search results to ir_measures run format: {qid: {did: score, ...}}
    """
    run = {}
    for qid, res in zip(query_ids, results):
        run[str(qid)] = {str(entry['corpus_id']): entry['score'] for entry in res}
    return run

def evaluate_results(
    results: List[List[Dict]],
    qrels: Dict[str, Dict[str, float]],
    query_ids: List[str],
) -> Dict:
    """
    Evaluate search results using ir_measures and print metrics.
    Returns the metrics dictionary.
    """
    run = convert_results_to_run(results, query_ids)
    metrics = ir_measures.calc_aggregate([nDCG@10, RR@10, RR@100, R@10, R@100, Success@5], qrels, run)
    print("\nEvaluation Metrics:", metrics)
    return metrics

def write_evaluation_metrics(
    filepath: Path,
    metrics: Dict
):
    """
    Write evaluation metrics to a JSON file.
    """
    metrics_str = {str(k): v for k, v in metrics.items()}  # Convert keys to strings for JSON serialization
    with open(filepath, "w") as f:
        json.dump(metrics_str, f, indent=4)
    print(f"\nEvaluation metrics written to {filepath}")

def read_jsonl(path: str) -> Tuple[List[str], List]:
    """
    Read encoded sparse embeddings from JSONL file.
    Returns: (ids, embeddings_decoded)
    Each embedding is a list of (token, weight) tuples.
    """
    with open(path, 'r') as f:
        embs_decoded = []
        ids = []
        for line in tqdm(f, desc=f"Reading JSONL {Path(path).name}"):
            data = orjson.loads(line)
            # Convert dict back to list of tuples for compatibility with Seismic
            embs_decoded.append([(token, weight) for token, weight in data['vector'].items()])
            ids.append(str(data['id']))  # Ensure IDs are strings
    return ids, embs_decoded

def read_jsonl_batched(path: str, batch_size: int = 50000):
    """
    Generator that yields batches of (ids, embeddings) from JSONL file.
    Yields: (batch_ids, batch_embeddings_decoded)
    """
    batch_ids = []
    batch_embs = []
    
    with open(path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc=f"Reading JSONL {Path(path).name}")):
            data = orjson.loads(line)
            batch_embs.append([(token, weight) for token, weight in data['vector'].items()])
            batch_ids.append(str(data['id']))
            
            if len(batch_ids) >= batch_size:
                yield batch_ids, batch_embs
                batch_ids = []
                batch_embs = []
        
        # Yield remaining items
        if batch_ids:
            yield batch_ids, batch_embs

def load_documents_from_parquet(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Load documents from a parquet file.
    Returns: (doc_ids, texts)
    """
    df = pd.read_parquet(filepath)
    doc_ids = df['docid'].tolist()
    texts = df['text'].tolist()
    return doc_ids, texts

def load_queries_from_jsonl(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Load queries from a jsonl file.
    Returns: (query_ids, texts)
    """
    query_ids = []
    texts = []
    with open(filepath, 'r') as f:
        for line in f:
            data = orjson.loads(line)
            query_ids.append(data['query_id'])
            texts.append(data['text'])
    return query_ids, texts

def load_qrels_from_file(filepath: str) -> Dict[str, Dict[str, int]]:
    """
    Load qrels from a file.
    Format: queryid 0 docid relscore
    Returns: {query_id: {doc_id: relevance}}
    """
    qrels = defaultdict(dict)
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])  # Convert to int instead of float
                qrels[query_id][doc_id] = relevance
    return dict(qrels)

def merge_qrels(qrels_list: List[Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    """
    Merge multiple qrels dictionaries into one.
    """
    merged = defaultdict(dict)
    for qrels in qrels_list:
        for qid, docs in qrels.items():
            merged[qid].update(docs)
    return dict(merged)

def discover_lotte_files(data_dir: Path) -> Dict[str, Dict]:
    """
    Discover LOTTE files organized by stream and session.
    Returns: {
        'documents': {stream: {session: filepath}},
        'queries': {stream: {session: filepath}},
        'qrels': {stream: {session: filepath}}
    }
    """
    discovered = {
        'documents': defaultdict(dict),
        'queries': defaultdict(dict),
        'qrels': defaultdict(dict)
    }
    
    docs_dir = data_dir / "docs"
    queries_dir = data_dir / "queries"
    qrels_dir = data_dir / "qrels"
    
    # Pattern: test_D{stream}_{session}.{ext}
    for stream in range(1,4):  # D1-D3
        for session in range(5):  # 0-4
            # Documents (parquet)
            doc_file = docs_dir / f"test_D{stream}_{session}.parquet"
            if doc_file.exists():
                discovered['documents'][stream][session] = str(doc_file)
            
            # Queries (jsonl)
            query_file = queries_dir / f"test_D{stream}_{session}.jsonl"
            if query_file.exists():
                discovered['queries'][stream][session] = str(query_file)
            
            # Qrels
            qrel_file = qrels_dir / f"test_D{stream}_{session}.qrels"
            if qrel_file.exists():
                discovered['qrels'][stream][session] = str(qrel_file)
                
                
    #example of the nested dictionary structure, 
    # {
    # 'documents': {
    #     1: {0: 'test_D1_0.parquet', 1: 'test_D1_1.parquet', ...},  # Stream D1
    #     2: {0: 'test_D2_0.parquet', 1: 'test_D2_1.parquet', ...},  # Stream D2
    #     3: {...}   # Stream D3
    # },
    # 'queries': { ... },  # Same structure
    # 'qrels': { ... }     # Same structure
    # }


    return {
        'documents': dict(discovered['documents']),
        'queries': dict(discovered['queries']),
        'qrels': dict(discovered['qrels'])
    }

def load_lotte_queries_and_qrels(dataset_name: str = "lotte/lifestyle/test"):
    """
    Load LOTTE queries and qrels for evaluation.
    Returns: queries dict, qrels dict
    """
    ds = ir_datasets.load(dataset_name)
    
    queries = {str(q.query_id): q.text for q in ds.queries_iter()}
    
    qrels = defaultdict(dict)
    for qrel in ds.qrels_iter():
        qrels[str(qrel.query_id)][str(qrel.doc_id)] = float(qrel.relevance)
    
    return queries, dict(qrels)

# def encode_stream_documents(
#     model: SparseEncoder,
#     documents: List[Dict],
#     batch_size: int,
#     output_jsonl: str,
#     chunk_size: int = 10000,
# ):
#     """
#     Encode documents from a stream and write to JSONL.
#     """
#     texts = [doc['text'] for doc in documents]
#     doc_ids = [doc['doc_id'] for doc in documents]
    
#     encode_and_save_to_jsonl(
#         model=model,
#         texts=texts,
#         ids=doc_ids,
#         output_jsonl=output_jsonl,
#         batch_size=batch_size,
#         chunk_size=chunk_size,
#         is_query=False,
#         description=f"Encoding documents to {Path(output_jsonl).name}"
#     )

def merge_jsonl_files(file_list: List[str], output_file: str):
    """
    Merge multiple JSONL files into one.
    """
    with open(output_file, 'w') as outf:
        for filepath in file_list:
            with open(filepath, 'r') as inf:
                for line in inf:
                    outf.write(line)

def extract_stream_from_docid(doc_id: str) -> int:
    """
    Extract stream number from document ID.
    E.g., 'test_D1_0' -> 1, 'test_D2_3' -> 2
    """
    parts = doc_id.split('_')
    if len(parts) >= 2 and parts[1].startswith('D'):
        return int(parts[1][1:])
    return None

def extract_session_from_docid(doc_id: str) -> int:
    """
    Extract session number from document ID.
    E.g., 'test_D1_0' -> 0, 'test_D2_3' -> 3
    """
    parts = doc_id.split('_')
    if len(parts) >= 3:
        return int(parts[2])
    return None

# def organize_documents_by_stream_session(dataset) -> Dict[int, Dict[int, List[Dict]]]:
#     """
#     Organize documents by stream and session extracted from doc_id.
#     Returns: {stream_num: {session_num: [doc_dicts]}}
#     """
#     organized = defaultdict(lambda: defaultdict(list))
    
#     for item in dataset:
#         doc_id = item['doc_id']
#         text = item['text']
        
#         stream = extract_stream_from_docid(doc_id)
#         session = extract_session_from_docid(doc_id)
        
#         if stream is not None and session is not None:
#             organized[stream][session].append({
#                 'doc_id': doc_id,
#                 'text': text
#             })
    
#     return {s: dict(d) for s, d in organized.items()}

# def organize_queries_by_stream(queries: Dict[str, str], qrels: Dict[str, Dict[str, float]]) -> Dict[int, List[str]]:
#     """
#     Organize queries by stream based on which documents they have relevance judgments for.
#     Returns: {stream_num: [query_ids]}
#     """
#     query_streams = defaultdict(set)
    
#     for qid, rels in qrels.items():
#         for doc_id in rels.keys():
#             stream = extract_stream_from_docid(doc_id)
#             if stream is not None:
#                 query_streams[qid].add(stream)
    
#     # Group queries by their primary stream (earliest stream with relevant docs)
#     streams_to_queries = defaultdict(list)
#     for qid in queries.keys():
#         if qid in query_streams and query_streams[qid]:
#             primary_stream = min(query_streams[qid])
#             streams_to_queries[primary_stream].append(qid)
    
#     return dict(streams_to_queries)

def pre_encode_all_data(
    data_files: Dict,
    old_model: SparseEncoder,
    new_model: SparseEncoder,
    output_dir: Path,
    args,
):
    """
    Pre-encode all documents and queries with both models and save to JSONL files.
    Files are organized by stream, session, and model matching the original structure.
    Returns metadata mapping to encoded files.
    """
    encoded_data = {
        'queries': {},  # {stream: {session: {model_name: path}}}
        'documents': {},  # {stream: {session: {model_name: path}}}
        'qrels': {}  # {stream: {session: path}}
    }
    
    # Create pre-encoding directory
    preenc_dir = output_dir / "pre_encoded"
    preenc_dir.mkdir(parents=True, exist_ok=True)
    
    # Print model information
    print(f"\nUsing encoders:")
    print(f"  Old model: {args.old_model}")
    print(f"  New model: {args.new_model}")
    
    # Pre-encode documents by stream and session
    print("\n" + "="*80)
    print("Pre-encoding documents by stream and session")
    print("="*80)
    
    for stream in sorted(data_files['documents'].keys()):
        print(f"\n--- Pre-encoding Documents Stream {stream} ---")
        encoded_data['documents'][stream] = {}
        
        for session in sorted(data_files['documents'][stream].keys()):
            doc_file = data_files['documents'][stream][session]
            print(f"  Session {session}: Loading from {Path(doc_file).name}")
            
            # Load documents
            doc_ids, texts = load_documents_from_parquet(doc_file)
            print(f"    Loaded {len(doc_ids)} documents")
            
            encoded_data['documents'][stream][session] = {}
            
            # Encode with both models
            for model_name, model in [('old', old_model), ('new', new_model)]:
                actual_model_name = args.old_model if model_name == 'old' else args.new_model
                output_jsonl = str(preenc_dir / f"docs_D{stream}_{session}_{model_name}.jsonl")
                print(f"    Encoding with {model_name} model ({actual_model_name})...")
                
                encode_and_save_to_jsonl(
                    model=model,
                    texts=texts,
                    ids=doc_ids,
                    output_jsonl=output_jsonl,
                    batch_size=32,
                    chunk_size=args.chunk_size,
                    is_query=False,
                    description=f"Docs D{stream}_{session} ({actual_model_name})"
                )
                
                encoded_data['documents'][stream][session][model_name] = output_jsonl
                
                # Clean up memory
                torch.cuda.empty_cache()
                gc.collect()
    
    # Pre-encode queries by stream and session
    print("\n" + "="*80)
    print("Pre-encoding queries by stream and session")
    print("="*80)
    
    for stream in sorted(data_files['queries'].keys()):
        print(f"\n--- Pre-encoding Queries Stream {stream} ---")
        encoded_data['queries'][stream] = {}
        
        for session in sorted(data_files['queries'][stream].keys()):
            query_file = data_files['queries'][stream][session]
            print(f"  Session {session}: Loading from {Path(query_file).name}")
            
            # Load queries
            query_ids, texts = load_queries_from_jsonl(query_file)
            print(f"    Loaded {len(query_ids)} queries")
            
            encoded_data['queries'][stream][session] = {}
            
            # Encode with both models
            for model_name, model in [('old', old_model), ('new', new_model)]:
                actual_model_name = args.old_model if model_name == 'old' else args.new_model
                output_jsonl = str(preenc_dir / f"queries_D{stream}_{session}_{model_name}.jsonl")
                print(f"    Encoding with {model_name} model ({actual_model_name})...")
                
                encode_and_save_to_jsonl(
                    model=model,
                    texts=texts,
                    ids=query_ids,
                    output_jsonl=output_jsonl,
                    batch_size=32,
                    chunk_size=len(texts),
                    is_query=True,
                    description=f"Queries D{stream}_{session} ({actual_model_name})"
                )
                
                encoded_data['queries'][stream][session][model_name] = output_jsonl
                
                # Clean up memory
                torch.cuda.empty_cache()
                gc.collect()
    
    # Copy qrels file paths
    print("\n" + "="*80)
    print("Recording qrels file paths")
    print("="*80)
    
    for stream in sorted(data_files['qrels'].keys()):
        encoded_data['qrels'][stream] = {}
        for session in sorted(data_files['qrels'][stream].keys()):
            encoded_data['qrels'][stream][session] = data_files['qrels'][stream][session]
            print(f"  Stream {stream}, Session {session}: {Path(data_files['qrels'][stream][session]).name}")
    
    # Save metadata
    metadata_file = preenc_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(encoded_data, f, indent=4)
    
    print(f"\nPre-encoding complete. Metadata saved to {metadata_file}")
    return encoded_data

def encode_missing_file_on_the_fly(
    model_name: str,
    model_path: str,
    original_file: str,
    output_file: str,
    file_type: str,
    stream: int,
    session: int,
) -> bool:
    """
    Encode a file on-the-fly when pre-encoded version is missing.
    
    Args:
        model_name: Label/key for the model (e.g., "old_v3doc")
        model_path: Actual model path (e.g., "naver/splade-v3-doc")
        original_file: Path to original data file (parquet/jsonl)
        output_file: Where to save the encoded file
        file_type: 'documents' or 'queries'
        stream: Stream number
        session: Session number
    
    Returns:
        True if encoding succeeded, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"ON-THE-FLY ENCODING: {model_name} for {file_type} stream {stream} session {session}")
    print(f"  Model: {model_path}")
    print(f"  Source: {original_file}")
    print(f"  Output: {output_file}")
    print(f"{'='*80}\n")
    
    try:
        # Load the model
        print(f"  Loading model: {model_path}")
        model = SparseEncoder(model_path)
        print(f"  Model loaded successfully")
        
        # Load the data
        if file_type == 'documents':
            print(f"  Loading documents from parquet...")
            ids, texts = load_documents_from_parquet(original_file)
            is_query = False
        elif file_type == 'queries':
            print(f"  Loading queries from jsonl...")
            ids, texts = load_queries_from_jsonl(original_file)
            is_query = True
        else:
            raise ValueError(f"Unknown file_type: {file_type}")
        
        print(f"  Loaded {len(ids)} {file_type}")
        
        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Encode and save
        encode_and_save_to_jsonl(
            model=model,
            texts=texts,
            ids=ids,
            output_jsonl=output_file,
            batch_size=32,
            chunk_size=10000,
            is_query=is_query,
            description=f"On-the-fly {file_type} D{stream}_{session} ({model_name})"
        )
        
        print(f"  ✓ Successfully encoded and saved to: {output_file}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"  ✗ ERROR: Failed to encode on-the-fly: {e}")
        import traceback
        traceback.print_exc()
        return False

def ensure_query_file_exists(
    query_files: Dict[str, str],
    model_key: str,
    model_paths: Dict[str, str],
    data_files: Dict,
    stream_num: int,
    session_num: int,
) -> str:
    """
    Ensure a query encoding file exists. If missing, encode on-the-fly.
    
    Returns:
        Path to the query file (existing or newly created)
    """
    if model_key in query_files:
        return query_files[model_key]
    
    print(f"  WARNING: '{model_key}' query encoding not found")
    print(f"  Available keys: {list(query_files.keys())}")
    
    # Get original data file
    original_query_file = data_files['queries'][stream_num][session_num]
    
    # Construct expected output path
    if query_files:
        output_dir = Path(list(query_files.values())[0]).parent
    else:
        # Fallback: create queries directory
        output_dir = Path("/ivi/ilps/scratch/ct/pre_encoded_murr/splade-v3-distilbert_splade-v3/queries")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = str(output_dir / f"queries_D{stream_num}_{session_num}_{model_key}.jsonl")
    
    # Get model path
    if model_key in model_paths:
        model_path = model_paths[model_key]
        print(f"  Attempting on-the-fly encoding with model: {model_path}")
        
        success = encode_missing_file_on_the_fly(
            model_name=model_key,
            model_path=model_path,
            original_file=original_query_file,
            output_file=output_file,
            file_type='queries',
            stream=stream_num,
            session=session_num,
        )
        
        if success:
            # Update dict with new file
            query_files[model_key] = output_file
            return output_file
        else:
            raise KeyError(f"Failed to encode {model_key} queries on-the-fly")
    else:
        raise KeyError(f"No model path defined for '{model_key}'. Add to MODEL_PATHS dict.")

def compute_success_at_k(
    results: List[List[Dict]],
    qrels: Dict[str, Dict[str, int]],
    query_ids: List[str],
    k: int = 5
) -> Dict[str, float]:
    """
    Compute Success@k for each query using ir_measures.
    Returns: {query_id: success_value (0.0 or 1.0)}
    """
    # Convert results to ir_measures run format
    run = convert_results_to_run(results, query_ids)
    
    # Compute Success@k for each query individually
    metric = Success @ k
    success_per_query = {}
    
    for qid in query_ids:
        qid_str = str(qid)
        if qid_str in qrels and qid_str in run:
            # Create single-query qrels and run
            single_qrels = {qid_str: qrels[qid_str]}
            single_run = {qid_str: run[qid_str]}
            
            # Compute metric
            result = ir_measures.calc_aggregate([metric], single_qrels, single_run)
            success_per_query[qid_str] = float(result[metric])
        else:
            success_per_query[qid_str] = 0.0
    
    return success_per_query

def compute_session_success_metrics(
    session_results: Dict[int, Dict],
    queries_by_session: Dict[int, List[str]],
) -> Dict:
    """
    Compute Success@5 metrics across sessions within a stream.
    
    Args:
        session_results: {session_num: {'success_per_query': {qid: 0/1}, ...}}
        queries_by_session: {session_num: [query_ids introduced in that session]}
    
    Returns:
        Dictionary with macro-average and per-query-session Success@5 metrics
    """
    S = max(session_results.keys()) + 1  # Number of sessions
    
    # 1) Per-query-session Success@5: Success@5(Q_i | IndexesUpTo(s))
    per_query_session_success = {}  # {(i, s): success_value}
    
    for eval_session_s in sorted(session_results.keys()):
        success_per_query_s = session_results[eval_session_s]['success_per_query']
        
        for query_session_i in sorted(queries_by_session.keys()):
            if query_session_i > eval_session_s:
                continue  # Can't evaluate queries not yet introduced
            
            queries_i = queries_by_session[query_session_i]
            # Get success values for queries from session i evaluated at session s
            success_values = [success_per_query_s.get(qid, 0) for qid in queries_i if qid in success_per_query_s]
            
            if success_values:
                per_query_session_success[(query_session_i, eval_session_s)] = np.mean(success_values)
            else:
                per_query_session_success[(query_session_i, eval_session_s)] = 0.0
    
    # 2) Macro-average Success@5 per evaluation session
    macro_success_per_session = {}
    
    for eval_session_s in sorted(session_results.keys()):
        success_per_query_s = session_results[eval_session_s]['success_per_query']
        # All queries up to and including session s
        all_queries_up_to_s = []
        for i in range(eval_session_s + 1):
            if i in queries_by_session:
                all_queries_up_to_s.extend(queries_by_session[i])
        
        success_values = [success_per_query_s.get(qid, 0) for qid in all_queries_up_to_s if qid in success_per_query_s]
        macro_success_per_session[eval_session_s] = np.mean(success_values) if success_values else 0.0
    
    # 3) Relative Success@5 gains
    gains = []
    
    for query_session_i in sorted(queries_by_session.keys()):
        for eval_session_s in sorted(session_results.keys()):
            if eval_session_s <= query_session_i:
                continue  # Need s > i for gain computation
            
            old_success = per_query_session_success.get((query_session_i, eval_session_s - 1), 0.0)
            new_success = per_query_session_success.get((query_session_i, eval_session_s), 0.0)
            gain = new_success - old_success
            gains.append({
                'query_session': query_session_i,
                'eval_session': eval_session_s,
                'old_success': old_success,
                'new_success': new_success,
                'gain': gain
            })
    
    mean_gain = np.mean([g['gain'] for g in gains]) if gains else 0.0
    std_gain = np.std([g['gain'] for g in gains]) if gains else 0.0
    
    return {
        'per_query_session_success': per_query_session_success,  # {(i, s): value}
        'macro_success_per_session': macro_success_per_session,  # {s: value}
        'relative_gains': gains,  # List of gain dicts
        'mean_gain': float(mean_gain),
        'std_gain': float(std_gain),
        'num_gain_pairs': len(gains)
    }

def compute_paired_ttest(
    baseline_a_results: Dict[int, Dict],
    baseline_b_results: Dict[int, Dict],
    queries_by_session: Dict[int, List[str]],
) -> Dict:
    """
    Compute paired t-test comparing two baselines on Success@5(Q_i | IndexesUpTo(s)).
    
    Returns:
        Dictionary with t-statistic and p-value
    """
    # Collect paired Success@5 values
    a_values = []
    b_values = []
    
    streams = sorted(set(baseline_a_results.keys()) & set(baseline_b_results.keys()))
    
    for eval_session_s in streams:
        success_a = baseline_a_results[eval_session_s]['success_per_query']
        success_b = baseline_b_results[eval_session_s]['success_per_query']
        
        for query_session_i in sorted(queries_by_session.keys()):
            if query_session_i > eval_session_s:
                continue
            
            queries_i = queries_by_session[query_session_i]
            
            # Compute Success@5(Q_i | IndexesUpTo(s)) for both strategies
            success_a_i_s = np.mean([success_a.get(qid, 0) for qid in queries_i if qid in success_a])
            success_b_i_s = np.mean([success_b.get(qid, 0) for qid in queries_i if qid in success_b])
            
            a_values.append(success_a_i_s)
            b_values.append(success_b_i_s)
    
    if len(a_values) > 1:
        t_stat, p_value = stats.ttest_rel(a_values, b_values)
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'n_pairs': len(a_values),
            'significant_at_0.05': p_value < 0.05
        }
    else:
        return {
            't_statistic': None,
            'p_value': None,
            'n_pairs': len(a_values),
            'significant_at_0.05': False,
            'error': 'Insufficient data for t-test'
        }

def run_streaming_baseline(
    baseline_config: Dict,
    encoded_data: Dict,
    data_files: Dict,
    output_dir: Path,
    args,
):
    """
    Run a streaming baseline experiment using pre-encoded embeddings.
    Processes each stream independently with incremental index building.
    If args.stream is specified, only processes that stream.
    """
    baseline_name = baseline_config['name']
    print(f"\n{'='*80}")
    print(f"Running Baseline: {baseline_name}")
    print(f"{'='*80}\n")
    
    # Using ephemeral positional mapping (no persistent file)
    
    # Track validation results for all streams and sessions
    all_validation_results = []
    
    # Determine document and query strategies
    doc_strategy = baseline_config['doc_model_strategy']
    query_strategy = baseline_config.get('query_model_strategy', baseline_config.get('query_model'))
    
    # Get actual model names from baseline config
    old_model_name = baseline_config.get('old_model_name', args.old_model)
    new_model_name = baseline_config.get('new_model_name', args.new_model)
    
    print(f"Old model: {old_model_name}")
    print(f"New model: {new_model_name}")
    
    # Check if rank fusion is enabled
    use_rank_fusion = baseline_config.get('use_rank_fusion', False)
    
    # Check if multiple indexes should be used
    use_multiple_indexes = args.use_multiple_indexes
    merge_method = args.merge_method if use_multiple_indexes else None
    
    if use_multiple_indexes:
        print(f"Using MULTIPLE indexes (one per session) with merge method: {merge_method}")
        if use_rank_fusion:
            print("WARNING: Rank fusion is not compatible with multiple indexes mode. Disabling rank fusion.")
            use_rank_fusion = False
    else:
        print(f"Using SINGLE cumulative index (standard approach)")
    
    # Determine which streams to process
    if hasattr(args, 'stream') and args.stream is not None:
        # Process only the specified stream (for SLURM job arrays)
        streams_to_process = [args.stream]
        print(f"Processing single stream: {args.stream}")
    else:
        # Process all streams
        streams_to_process = sorted(encoded_data['documents'].keys())
        print(f"Processing all streams: {streams_to_process}")
    
    # Process each stream independently
    all_streams_results = {}
    
    for stream_num in streams_to_process:
        # Validate stream exists
        if stream_num not in encoded_data['documents']:
            print(f"Warning: Stream {stream_num} not found in encoded data, skipping...")
            continue
            
        print(f"\n{'='*80}")
        print(f"Processing Stream {stream_num}")
        print(f"{'='*80}\n")
        
        stream_dir = output_dir / baseline_name / f"stream_{stream_num}"
        stream_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cumulative document IDs list for this stream (ephemeral, in-memory only)
        # Maps positional indices (0, 1, 2, ...) to actual document IDs
        cumulative_doc_ids = []
        
        # Track results and queries for this stream
        stream_results_per_session = {}
        stream_queries_by_session = {}
        
        # Track cumulative document JSONL files (with numeric IDs) for index building
        cumulative_doc_jsonl_files = []
        
        # Track session indexes if using multiple indexes approach
        session_indexes = [] if use_multiple_indexes else None
        
        # Process each session within this stream
        for session_num in sorted(encoded_data['documents'][stream_num].keys()):
            print(f"\n--- Stream {stream_num}, Session {session_num} ---")
            print(f"{'='*80}")
            
            session_dir = stream_dir / f"session_{session_num}"
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # ========================================
            # PART 1: Determine which model to use for documents
            # ========================================
            session_files = encoded_data['documents'][stream_num][session_num]
            
            # Define model paths for on-the-fly encoding
            MODEL_PATHS = {
                'old': 'rasyosef/splade-tiny',
                'old_tiny': 'rasyosef/splade-tiny',
                'old_v3doc': 'naver/splade-v3-doc',
                'old_v3lexical': 'naver/splade-v3-lexical',
                'old_v3distilbert': 'naver/splade-v3-distilbert',
                'old_cosd': 'naver/splade-cocondenser-selfdistil',
                'new': 'naver/splade-v3',
            }
            
            if doc_strategy == 'old':
                doc_model_key = old_model_name
                doc_model_used_name = old_model_name
            elif doc_strategy == 'new':
                doc_model_key = new_model_name
                doc_model_used_name = new_model_name
            elif doc_strategy == 'switch':
                if session_num >= baseline_config.get('switch_session', 2):
                    doc_model_key = new_model_name
                    doc_model_used_name = new_model_name
                else:
                    doc_model_key = old_model_name
                    doc_model_used_name = old_model_name
            else:
                raise ValueError(f"Unknown doc_model_strategy: {doc_strategy}")
            
            # Check if file exists in metadata AND on disk, if not encode on-the-fly
            if doc_model_key not in session_files:
                print(f"  WARNING: '{doc_model_key}' encoding not found in metadata")
                print(f"  Available keys: {list(session_files.keys())}")
                needs_encoding = True
                doc_file = None
            else:
                doc_file = session_files[doc_model_key]
                # Check if file actually exists on disk
                if not Path(doc_file).exists():
                    print(f"  WARNING: '{doc_model_key}' file path in metadata but file doesn't exist:")
                    print(f"    Path: {doc_file}")
                    needs_encoding = True
                else:
                    needs_encoding = False
            
            if needs_encoding:
                # Get original data file
                original_doc_file = data_files['documents'][stream_num][session_num]
                
                # Construct expected output path
                if doc_file:
                    # Use path from metadata
                    output_file = doc_file
                else:
                    # Construct new path
                    output_dir_path = Path(list(session_files.values())[0]).parent  # Use same dir as other encodings
                    output_file = str(output_dir_path / f"docs_D{stream_num}_{session_num}_{doc_model_key}.jsonl")
                
                # Get model path
                if doc_model_key in MODEL_PATHS:
                    model_path = MODEL_PATHS[doc_model_key]
                    print(f"  Attempting on-the-fly encoding with model: {model_path}")
                    
                    success = encode_missing_file_on_the_fly(
                        model_name=doc_model_key,
                        model_path=model_path,
                        original_file=original_doc_file,
                        output_file=output_file,
                        file_type='documents',
                        stream=stream_num,
                        session=session_num,
                    )
                    
                    if success:
                        # Update metadata with new file
                        session_files[doc_model_key] = output_file
                        doc_file = output_file
                    else:
                        raise KeyError(f"Failed to encode {doc_model_key} on-the-fly")
                else:
                    raise KeyError(f"No model path defined for '{doc_model_key}'. Add to MODEL_PATHS dict.")
            
            print(f"  Using document encoder: {doc_model_used_name}")
            print(f"  Using document file: {Path(doc_file).name}")
            
            # Determine which model is being used for documents (for validation)
            if doc_strategy == 'old':
                doc_model_used = 'old'
            elif doc_strategy == 'new':
                doc_model_used = 'new'
            elif doc_strategy == 'switch':
                doc_model_used = 'new' if session_num >= baseline_config.get('switch_session', 2) else 'old'
            
            # ========================================
            # PART 2: Convert documents to numeric IDs
            # ========================================
            print(f"  Converting document IDs to numeric...")
            temp_doc_jsonl = str(session_dir / f"temp_docs_numeric_session{session_num}.jsonl")
            
            # Track loaded IDs for validation
            session_doc_ids_loaded = []
            
            total_docs_written = 0
            try:
                with open(temp_doc_jsonl, 'w', buffering=1024*1024) as outf:  # Add buffer
                    for batch_ids, batch_embs in read_jsonl_batched(doc_file, batch_size=args.batch_size):
                        # Assign sequential positional IDs and track string IDs
                        batch_start_id = len(cumulative_doc_ids)
                        cumulative_doc_ids.extend(batch_ids)
                        
                        # Track IDs for validation
                        session_doc_ids_loaded.extend(batch_ids)
                        
                        # Write with positional IDs (0, 1, 2, ...)
                        for idx, emb in enumerate(batch_embs):
                            positional_id = batch_start_id + idx
                            sparse_vector = {token: float(weight) for token, weight in emb}
                            json.dump({'id': positional_id, 'vector': sparse_vector}, outf)
                            outf.write('\n')
                        
                        total_docs_written += len(batch_ids)
                        del batch_ids, batch_embs
                        gc.collect()
                
                # Verify file was written and is not empty
                file_size = Path(temp_doc_jsonl).stat().st_size
                if file_size == 0:
                    raise ValueError(f"Converted JSONL file is empty! Expected {total_docs_written} documents.")
                
                print(f"  Converted and verified {total_docs_written} documents (file size: {file_size} bytes)")
                
                # Validate loaded document IDs against original data if available
                if data_files and stream_num in data_files['documents'] and session_num in data_files['documents'][stream_num]:
                    original_doc_file = data_files['documents'][stream_num][session_num]
                    print(f"\n  [VALIDATION] Checking document IDs against original data...")
                    doc_validation = validate_ids_against_original(
                        loaded_ids=session_doc_ids_loaded,
                        original_file=original_doc_file,
                        file_type='documents',
                        stream=stream_num,
                        session=session_num,
                        model_used=doc_model_used
                    )
                    all_validation_results.append(doc_validation)
                    
                    if doc_validation['perfect_match']:
                        print(f"  ✓ Document IDs: {doc_validation['loaded_count']}/{doc_validation['original_count']} (100% match, model={doc_model_used_name})")
                    else:
                        print(f"  ✗ Document IDs: {doc_validation['loaded_count']}/{doc_validation['original_count']} "
                              f"({doc_validation['match_percentage']:.2f}% match, model={doc_model_used_name})")
                        if doc_validation['missing_ids_count'] > 0:
                            print(f"    Missing {doc_validation['missing_ids_count']} IDs")
                        if doc_validation['extra_ids_count'] > 0:
                            print(f"    Extra {doc_validation['extra_ids_count']} IDs")
                else:
                    print(f"\n  [VALIDATION] Skipping document ID validation (original data not available)")

            except Exception as e:
                # Clean up corrupted file
                if Path(temp_doc_jsonl).exists():
                    Path(temp_doc_jsonl).unlink()
                print(f"  ERROR during document conversion: {e}")
                raise
            
            # Add this session's JSONL to cumulative list
            cumulative_doc_jsonl_files.append(temp_doc_jsonl)
            
            # ========================================
            # PART 3: Build index(es) - either single cumulative or per-session
            # ========================================
            if use_multiple_indexes:
                # Build a separate index for this session only
                print(f"\n  Building index for session {session_num} (multiple indexes mode)...")
                
                # Verify session file exists
                if not Path(temp_doc_jsonl).exists():
                    raise FileNotFoundError(f"Session file missing: {temp_doc_jsonl}")
                session_file_size = Path(temp_doc_jsonl).stat().st_size
                if session_file_size == 0:
                    raise ValueError(f"Session file is empty: {temp_doc_jsonl}")
                print(f"  Session file size: {session_file_size} bytes")
                
                # Build session index
                print(f"  Building Seismic index for session {session_num}...")
                try:
                    session_index = SeismicIndex.build(temp_doc_jsonl)
                    session_indexes.append(session_index)
                    print(f"  Built index {len(session_indexes)} (session {session_num})")
                except Exception as e:
                    print(f"  ERROR building Seismic index for session {session_num}: {e}")
                    raise
                
                # Clean up temp JSONL file immediately after building index
                if Path(temp_doc_jsonl).exists():
                    Path(temp_doc_jsonl).unlink()
                    print(f"  Deleted temp file: {Path(temp_doc_jsonl).name}")
                else:
                    print(f"  Temp file already cleaned up by Seismic index builder")
                
            else:
                # Build single cumulative index from all sessions
                print(f"\n  Building cumulative index from {len(cumulative_doc_jsonl_files)} session(s)...")
                
                # Create merged JSONL file for all sessions up to current
                merged_jsonl = str(session_dir / f"merged_docs_stream{stream_num}_upto_session{session_num}.jsonl")
                print(f"  Merging JSONL files into: {Path(merged_jsonl).name}")
                
                # Verify files exist before merging
                for f in cumulative_doc_jsonl_files:
                    if not Path(f).exists():
                        raise FileNotFoundError(f"Input file missing: {f}")
                    f_size = Path(f).stat().st_size
                    if f_size == 0:
                        raise ValueError(f"Input file is empty: {f}")
                    print(f"    Input file {Path(f).name}: {f_size} bytes")
                
                merge_jsonl_files(cumulative_doc_jsonl_files, merged_jsonl)
                
                # Verify merged file
                merged_size = Path(merged_jsonl).stat().st_size
                if merged_size == 0:
                    raise ValueError(f"Merged JSONL file is empty! Check input files.")
                print(f"  Merged file size: {merged_size} bytes")
                
                # Validate JSONL format before indexing
                print(f"  Validating merged JSONL format...")
                line_count = 0
                with open(merged_jsonl, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if line_num % 100000 == 0:
                            print(f"    Validated {line_num} lines...")
                        try:
                            data = orjson.loads(line)
                            if 'id' not in data or 'vector' not in data:
                                raise ValueError(f"Invalid structure at line {line_num}")
                            if not data['vector']:  # Empty vector check
                                print(f"    WARNING: Empty vector at line {line_num}, id={data['id']}")
                            line_count += 1
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON at line {line_num}: {e}")
                
                if line_count == 0:
                    raise ValueError("No valid documents found in merged JSONL!")
                print(f"  JSONL validation complete: {line_count} valid documents")
                
                # Build index from merged JSONL (keep in memory, don't save)
                print(f"  Building Seismic index...")
                try:
                    index = SeismicIndex.build(merged_jsonl)
                except Exception as e:
                    print(f"  ERROR building Seismic index: {e}")
                    raise
                
                # Clean up merged JSONL file (if it still exists)
                if Path(merged_jsonl).exists():
                    Path(merged_jsonl).unlink()
                    print(f"  Deleted temporary merged file: {Path(merged_jsonl).name}")
                else:
                    print(f"  Merged file already cleaned up by Seismic index builder")
            
            # ========================================
            # PART 4: Load ALL cumulative queries with correct model
            # ========================================
            # Check if this baseline uses adapter/refined queries
            query_adapter = baseline_config.get('query_adapter', None)
            adapter_before_switch = baseline_config.get('adapter_before_switch', False)
            
            # Determine which query model should be used for this session
            if query_strategy == 'always_rep_fuse':
                # Baseline21: Always use BOTH old and new encodings with representation fusion
                current_query_model = 'new'  # Primary model (for display)
                current_query_model_name = f"{old_model_name} + {new_model_name} (always rep fused)"
            elif query_strategy == 'rep_fuse_before_switch':
                # Baseline22: Rep fusion BEFORE switch, then new-only AFTER switch
                if session_num < baseline_config.get('switch_session', 2):
                    current_query_model = 'new'  # Primary model (for display)
                    current_query_model_name = f"{old_model_name} + {new_model_name} (rep fused before switch)"
                else:
                    current_query_model = 'new'  # New model only after switch
                    current_query_model_name = f"{new_model_name} (after switch, no fusion)"
            elif query_strategy == 'rep_fuse_after_switch':
                # Baseline23: Old-only BEFORE switch, then rep fusion AFTER switch
                if session_num < baseline_config.get('switch_session', 2):
                    current_query_model = 'old'  # Old model only before switch
                    current_query_model_name = f"{old_model_name} (before switch, no fusion)"
                else:
                    current_query_model = 'new'  # Primary model for rep fusion after switch
                    current_query_model_name = f"{old_model_name} + {new_model_name} (rep fused after switch)"
            elif query_strategy == 'always_fuse':
                # Baseline15: Always use BOTH old and new encodings with fusion
                current_query_model = 'new'  # Primary model (for display)
                current_query_model_name = f"{old_model_name} + {new_model_name} (always fused)"
            elif query_strategy == 'fuse_before_switch':
                # Baseline16: Fuse BEFORE switch, then new-only AFTER switch
                if session_num < baseline_config.get('switch_session', 2):
                    current_query_model = 'new'  # Primary model for fusion
                    current_query_model_name = f"{old_model_name} + {new_model_name} (fused before switch)"
                else:
                    current_query_model = 'new'  # New model only after switch
                    current_query_model_name = f"{new_model_name} (after switch, no fusion)"
            elif query_strategy == 'fuse_after_switch':
                # Baseline17: Old-only BEFORE switch, then fusion AFTER switch
                if session_num < baseline_config.get('switch_session', 2):
                    current_query_model = 'old'  # Old model only before switch
                    current_query_model_name = f"{old_model_name} (before switch, no fusion)"
                else:
                    current_query_model = 'new'  # Primary model for fusion after switch
                    current_query_model_name = f"{old_model_name} + {new_model_name} (fused after switch)"
            elif query_strategy == 'switch':
                current_query_model = new_model_name if session_num >= baseline_config.get('switch_session', 2) else old_model_name
                current_query_model_name = new_model_name if session_num >= baseline_config.get('switch_session', 2) else old_model_name
                
                # If adapter is specified
                if query_adapter:
                    if adapter_before_switch:
                        # baseline14: Use adapter BEFORE switch, then switch to new model
                        if session_num < baseline_config.get('switch_session', 2):
                            current_query_model = query_adapter  # Use adapter before switch
                            current_query_model_name = f"{old_model_name} (refined: {query_adapter})"
                        else:
                            current_query_model = new_model_name  # Switch to new model
                            current_query_model_name = new_model_name
                    else:
                        # baseline11/baseline19: Use old model before switch, then adapter AFTER switch
                        if session_num >= baseline_config.get('switch_session', 2):
                            current_query_model = query_adapter  # Use adapter after switch
                            current_query_model_name = f"{old_model_name} (refined: {query_adapter})"
            elif query_strategy in ['old', 'new']:
                current_query_model = old_model_name if query_strategy == 'old' else new_model_name
                current_query_model_name = old_model_name if query_strategy == 'old' else new_model_name
            elif query_strategy in ['old_refined_tiny', 'new_refined_doc', 'new_refined_lexical']:
                # Baseline18: Always use adapter model
                current_query_model = query_strategy
                if query_strategy == 'old_refined_tiny':
                    current_query_model_name = f"{old_model_name} (adapter: old_refined_tiny)"
                elif query_strategy == 'new_refined_doc':
                    current_query_model_name = f"{new_model_name} (adapter: new_refined_doc)"
                elif query_strategy == 'new_refined_lexical':
                    current_query_model_name = f"{new_model_name} (adapter: new_refined_lexical)"
            else:
                raise ValueError(f"Unknown query_model_strategy: {query_strategy}")
            
            print(f"  Using query encoder: {current_query_model_name}")
            
            # Define model paths for on-the-fly encoding
            MODEL_PATHS = {
                'old': 'rasyosef/splade-tiny',
                'old_tiny': 'rasyosef/splade-tiny',
                'old_v3doc': 'naver/splade-v3-doc',
                'old_v3lexical': 'naver/splade-v3-lexical',
                'old_v3distilbert': 'naver/splade-v3-distilbert',
                'old_cosd': 'naver/splade-cocondenser-selfdistil',
                'new': 'naver/splade-v3',
            }
            
            # Reload ALL queries from session 0 to current session with the correct model
            cumulative_query_ids = []
            cumulative_query_embs = []
            
            # Initialize cumulative_query_embs_old if needed for rank fusion OR representation fusion
            needs_old_embs = use_rank_fusion or query_strategy in [
                'always_rep_fuse', 'rep_fuse_before_switch', 'rep_fuse_after_switch',
                'fuse_before_switch', 'fuse_after_switch', 'always_fuse'
            ]
            cumulative_query_embs_old = [] if needs_old_embs else None
            
            print(f"\n  [QUERY LOADING] Loading cumulative queries from session 0 to {session_num}...")
            
            for past_session in range(session_num + 1):
                if past_session in encoded_data['queries'][stream_num]:
                    query_files = encoded_data['queries'][stream_num][past_session]
                    
                    # Determine which query encoder to use for this specific session
                    if query_strategy == 'fuse_before_switch':
                        # Baseline16: Fuse before switch, new-only after
                        if session_num < baseline_config.get('switch_session', 2):
                            # BEFORE switch: Load both old and new for fusion
                            query_file_new = ensure_query_file_exists(query_files, new_model_name, MODEL_PATHS, data_files, stream_num, past_session)
                            query_file_old = ensure_query_file_exists(query_files, old_model_name, MODEL_PATHS, data_files, stream_num, past_session)
                            
                            print(f"  Loading NEW queries from session {past_session}: {Path(query_file_new).name}")
                            session_query_ids, session_query_embs_new = read_jsonl(query_file_new)
                            cumulative_query_ids.extend(session_query_ids)
                            cumulative_query_embs.extend(session_query_embs_new)
                            
                            print(f"  Loading OLD queries from session {past_session} for fusion: {Path(query_file_old).name}")
                            _, session_query_embs_old = read_jsonl(query_file_old)
                            cumulative_query_embs_old.extend(session_query_embs_old)
                        else:
                            # AFTER switch: Load only new, no fusion
                            query_file_new = ensure_query_file_exists(query_files, new_model_name, MODEL_PATHS, data_files, stream_num, past_session)
                            print(f"  ✗ ERROR: 'new' encoding required after switch at session {past_session}")
                            print(f"    Available: {list(query_files.keys())}")
                            raise KeyError(f"'new' encoding required for fuse_before_switch after session {baseline_config.get('switch_session')}")
                            
                            query_file_new = query_files[new_model_name]
                            print(f"  Loading NEW queries from session {past_session} (no fusion after switch): {Path(query_file_new).name}")
                            session_query_ids, session_query_embs_new = read_jsonl(query_file_new)
                            cumulative_query_ids.extend(session_query_ids)
                            cumulative_query_embs.extend(session_query_embs_new)
                            # No old embeddings loaded after switch
                        
                        # Validation
                        if data_files and stream_num in data_files['queries'] and past_session in data_files['queries'][stream_num]:
                            original_query_file = data_files['queries'][stream_num][past_session]
                            print(f"  [VALIDATION] Checking query IDs against original data...")
                            fusion_status = 'fused' if session_num < baseline_config.get('switch_session', 2) else 'new_only'
                            query_validation = validate_ids_against_original(
                                loaded_ids=session_query_ids,
                                original_file=original_query_file,
                                file_type='queries',
                                stream=stream_num,
                                session=past_session,
                                model_used=fusion_status
                            )
                            all_validation_results.append(query_validation)
                            
                            if query_validation['perfect_match']:
                                print(f"  ✓ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} (100% match, {fusion_status})")
                            else:
                                print(f"  ✗ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} "
                                      f"({query_validation['match_percentage']:.2f}% match, {fusion_status})")
                                if query_validation['missing_ids_count'] > 0:
                                    print(f"    Missing {query_validation['missing_ids_count']} IDs")
                                if query_validation['extra_ids_count'] > 0:
                                    print(f"    Extra {query_validation['extra_ids_count']} IDs")
                    
                    elif query_strategy == 'fuse_after_switch':
                        # Baseline17: Old-only before switch, fusion after
                        if session_num < baseline_config.get('switch_session', 2):
                            # BEFORE switch: Load only old, no fusion
                            if old_model_name not in query_files:
                                print(f"  ✗ ERROR: 'old' encoding required before switch at session {past_session}")
                                print(f"    Available: {list(query_files.keys())}")
                                raise KeyError(f"'old' encoding required for fuse_after_switch before session {baseline_config.get('switch_session')}")
                            
                            query_file_old = query_files[old_model_name]
                            print(f"  Loading OLD queries from session {past_session} (no fusion before switch): {Path(query_file_old).name}")
                            session_query_ids, session_query_embs_old = read_jsonl(query_file_old)
                            cumulative_query_ids.extend(session_query_ids)
                            cumulative_query_embs.extend(session_query_embs_old)
                            # No new embeddings loaded before switch
                        else:
                            # AFTER switch: Load both old and new for fusion
                            if new_model_name not in query_files or old_model_name not in query_files:
                                print(f"  ✗ ERROR: Both 'old' and 'new' encodings required for fusion at session {past_session}")
                                print(f"    Available: {list(query_files.keys())}")
                                raise KeyError(f"Both encodings required for fuse_after_switch after session {baseline_config.get('switch_session')}")
                            
                            query_file_new = query_files[new_model_name]
                            print(f"  Loading NEW queries from session {past_session}: {Path(query_file_new).name}")
                            session_query_ids, session_query_embs_new = read_jsonl(query_file_new)
                            cumulative_query_ids.extend(session_query_ids)
                            cumulative_query_embs.extend(session_query_embs_new)
                            
                            query_file_old = query_files[old_model_name]
                            print(f"  Loading OLD queries from session {past_session} for fusion: {Path(query_file_old).name}")
                            _, session_query_embs_old = read_jsonl(query_file_old)
                            cumulative_query_embs_old.extend(session_query_embs_old)
                        
                        # Validation
                        if data_files and stream_num in data_files['queries'] and past_session in data_files['queries'][stream_num]:
                            original_query_file = data_files['queries'][stream_num][past_session]
                            print(f"  [VALIDATION] Checking query IDs against original data...")
                            fusion_status = 'fused' if session_num >= baseline_config.get('switch_session', 2) else 'old_only'
                            query_validation = validate_ids_against_original(
                                loaded_ids=session_query_ids,
                                original_file=original_query_file,
                                file_type='queries',
                                stream=stream_num,
                                session=past_session,
                                model_used=fusion_status
                            )
                            all_validation_results.append(query_validation)
                            
                            if query_validation['perfect_match']:
                                print(f"  ✓ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} (100% match, {fusion_status})")
                            else:
                                print(f"  ✗ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} "
                                      f"({query_validation['match_percentage']:.2f}% match, {fusion_status})")
                                if query_validation['missing_ids_count'] > 0:
                                    print(f"    Missing {query_validation['missing_ids_count']} IDs")
                                if query_validation['extra_ids_count'] > 0:
                                    print(f"    Extra {query_validation['extra_ids_count']} IDs")
                    
                    elif query_strategy == 'rep_fuse_before_switch':
                        # Baseline22: Rep fusion BEFORE switch, new-only AFTER switch
                        if session_num < baseline_config.get('switch_session', 2):
                            # BEFORE switch: Load both old and new for representation fusion
                            if new_model_name not in query_files or old_model_name not in query_files:
                                print(f"  ✗ ERROR: Both 'old' and 'new' query encodings required for rep_fuse_before_switch before switch session")
                                print(f"    Available: {list(query_files.keys())}")
                                raise KeyError(f"Both encodings required for rep_fuse_before_switch before session {baseline_config.get('switch_session')}")
                            
                            query_file_new = query_files[new_model_name]
                            print(f"  Loading NEW queries from session {past_session}: {Path(query_file_new).name}")
                            session_query_ids, session_query_embs_new = read_jsonl(query_file_new)
                            cumulative_query_ids.extend(session_query_ids)
                            cumulative_query_embs.extend(session_query_embs_new)
                            
                            query_file_old = query_files[old_model_name]
                            print(f"  Loading OLD queries from session {past_session} for representation fusion: {Path(query_file_old).name}")
                            _, session_query_embs_old = read_jsonl(query_file_old)
                            cumulative_query_embs_old.extend(session_query_embs_old)
                        else:
                            # AFTER switch: Load only new, no fusion
                            if new_model_name not in query_files:
                                print(f"  ✗ ERROR: 'new' query encoding not found for stream {stream_num}, session {past_session}")
                                print(f"    Available: {list(query_files.keys())}")
                                raise KeyError(f"'new' encoding required for rep_fuse_before_switch after switch session")
                            
                            query_file_new = query_files[new_model_name]
                            print(f"  Loading NEW queries from session {past_session} (no fusion after switch): {Path(query_file_new).name}")
                            session_query_ids, session_query_embs_new = read_jsonl(query_file_new)
                            cumulative_query_ids.extend(session_query_ids)
                            cumulative_query_embs.extend(session_query_embs_new)
                            # No old embeddings loaded after switch
                        
                        # Validation
                        if data_files and stream_num in data_files['queries'] and past_session in data_files['queries'][stream_num]:
                            original_query_file = data_files['queries'][stream_num][past_session]
                            print(f"  [VALIDATION] Checking query IDs against original data...")
                            fusion_status = 'rep_fused' if session_num < baseline_config.get('switch_session', 2) else 'new_only'
                            query_validation = validate_ids_against_original(
                                loaded_ids=session_query_ids,
                                original_file=original_query_file,
                                file_type='queries',
                                stream=stream_num,
                                session=past_session,
                                model_used=fusion_status
                            )
                            all_validation_results.append(query_validation)
                            
                            if query_validation['perfect_match']:
                                print(f"  ✓ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} (100% match, {fusion_status})")
                            else:
                                print(f"  ✗ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} "
                                      f"({query_validation['match_percentage']:.2f}% match, {fusion_status})")
                                if query_validation['missing_ids_count'] > 0:
                                    print(f"    Missing {query_validation['missing_ids_count']} IDs")
                                if query_validation['extra_ids_count'] > 0:
                                    print(f"    Extra {query_validation['extra_ids_count']} IDs")
                    
                    elif query_strategy == 'rep_fuse_after_switch':
                        # Baseline23: Old-only BEFORE switch, rep fusion AFTER switch
                        if session_num < baseline_config.get('switch_session', 2):
                            # BEFORE switch: Load only old, no fusion
                            if old_model_name not in query_files:
                                print(f"  ✗ ERROR: 'old' query encoding not found for stream {stream_num}, session {past_session}")
                                print(f"    Available: {list(query_files.keys())}")
                                raise KeyError(f"'old' encoding required for rep_fuse_after_switch before switch session")
                            
                            query_file_old = query_files[old_model_name]
                            print(f"  Loading OLD queries from session {past_session} (no fusion before switch): {Path(query_file_old).name}")
                            session_query_ids, session_query_embs_old = read_jsonl(query_file_old)
                            cumulative_query_ids.extend(session_query_ids)
                            cumulative_query_embs.extend(session_query_embs_old)
                            # No new embeddings loaded before switch
                        else:
                            # AFTER switch: Load both old and new for representation fusion
                            if new_model_name not in query_files or old_model_name not in query_files:
                                print(f"  ✗ ERROR: Both 'old' and 'new' query encodings required for rep_fuse_after_switch after switch session")
                                print(f"    Available: {list(query_files.keys())}")
                                raise KeyError(f"Both encodings required for rep_fuse_after_switch after session {baseline_config.get('switch_session')}")
                            
                            query_file_new = query_files[new_model_name]
                            print(f"  Loading NEW queries from session {past_session}: {Path(query_file_new).name}")
                            session_query_ids, session_query_embs_new = read_jsonl(query_file_new)
                            cumulative_query_ids.extend(session_query_ids)
                            cumulative_query_embs.extend(session_query_embs_new)
                            
                            query_file_old = query_files[old_model_name]
                            print(f"  Loading OLD queries from session {past_session} for representation fusion: {Path(query_file_old).name}")
                            _, session_query_embs_old = read_jsonl(query_file_old)
                            cumulative_query_embs_old.extend(session_query_embs_old)
                        
                        # Validation
                        if data_files and stream_num in data_files['queries'] and past_session in data_files['queries'][stream_num]:
                            original_query_file = data_files['queries'][stream_num][past_session]
                            print(f"  [VALIDATION] Checking query IDs against original data...")
                            fusion_status = 'old_only' if session_num < baseline_config.get('switch_session', 2) else 'rep_fused'
                            query_validation = validate_ids_against_original(
                                loaded_ids=session_query_ids,
                                original_file=original_query_file,
                                file_type='queries',
                                stream=stream_num,
                                session=past_session,
                                model_used=fusion_status
                            )
                            all_validation_results.append(query_validation)
                            
                            if query_validation['perfect_match']:
                                print(f"  ✓ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} (100% match, {fusion_status})")
                            else:
                                print(f"  ✗ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} "
                                      f"({query_validation['match_percentage']:.2f}% match, {fusion_status})")
                                if query_validation['missing_ids_count'] > 0:
                                    print(f"    Missing {query_validation['missing_ids_count']} IDs")
                                if query_validation['extra_ids_count'] > 0:
                                    print(f"    Extra {query_validation['extra_ids_count']} IDs")
                    
                    elif query_strategy == 'always_rep_fuse':
                        # Baseline21: Always load BOTH old and new encodings for representation fusion
                        # Load new (primary) encoding
                        if new_model_name not in query_files:
                            print(f"  ✗ ERROR: 'new' query encoding not found for stream {stream_num}, session {past_session}")
                            print(f"    Available: {list(query_files.keys())}")
                            raise KeyError(f"'new' encoding required for always_rep_fuse strategy")
                        
                        query_file_new = query_files[new_model_name]
                        print(f"  Loading NEW queries from session {past_session}: {Path(query_file_new).name}")
                        session_query_ids, session_query_embs_new = read_jsonl(query_file_new)
                        
                        cumulative_query_ids.extend(session_query_ids)
                        cumulative_query_embs.extend(session_query_embs_new)
                        
                        # Load old encoding for representation fusion (MUST exist)
                        if old_model_name not in query_files:
                            print(f"  ✗ ERROR: 'old' query encoding not found for stream {stream_num}, session {past_session}")
                            print(f"    Available: {list(query_files.keys())}")
                            raise KeyError(f"'old' encoding required for always_rep_fuse strategy")
                        
                        query_file_old = query_files[old_model_name]
                        print(f"  Loading OLD queries from session {past_session} for representation fusion: {Path(query_file_old).name}")
                        _, session_query_embs_old = read_jsonl(query_file_old)
                        cumulative_query_embs_old.extend(session_query_embs_old)
                        
                        # Validate query IDs against original data if available
                        if data_files and stream_num in data_files['queries'] and past_session in data_files['queries'][stream_num]:
                            original_query_file = data_files['queries'][stream_num][past_session]
                            print(f"  [VALIDATION] Checking query IDs against original data...")
                            query_validation = validate_ids_against_original(
                                loaded_ids=session_query_ids,
                                original_file=original_query_file,
                                file_type='queries',
                                stream=stream_num,
                                session=past_session,
                                model_used='new+old (rep_fuse)'
                            )
                            all_validation_results.append(query_validation)
                            
                            if query_validation['perfect_match']:
                                print(f"  ✓ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} (100% match, rep fusion enabled)")
                            else:
                                print(f"  ✗ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} "
                                      f"({query_validation['match_percentage']:.2f}% match, rep fusion enabled)")
                                if query_validation['missing_ids_count'] > 0:
                                    print(f"    Missing {query_validation['missing_ids_count']} IDs")
                                if query_validation['extra_ids_count'] > 0:
                                    print(f"    Extra {query_validation['extra_ids_count']} IDs")
                    
                    elif query_strategy == 'always_fuse':
                        # Baseline15: Always load BOTH old and new encodings
                        # Load new (primary) encoding
                        if new_model_name not in query_files:
                            print(f"  ✗ ERROR: 'new' query encoding not found for stream {stream_num}, session {past_session}")
                            print(f"    Available: {list(query_files.keys())}")
                            raise KeyError(f"'new' encoding required for always_fuse strategy")
                        
                        query_file_new = query_files[new_model_name]
                        print(f"  Loading NEW queries from session {past_session}: {Path(query_file_new).name}")
                        session_query_ids, session_query_embs_new = read_jsonl(query_file_new)
                        
                        cumulative_query_ids.extend(session_query_ids)
                        cumulative_query_embs.extend(session_query_embs_new)
                        
                        # Load old encoding for fusion (MUST exist)
                        if old_model_name not in query_files:
                            print(f"  ✗ ERROR: 'old' query encoding not found for stream {stream_num}, session {past_session}")
                            print(f"    Available: {list(query_files.keys())}")
                            raise KeyError(f"'old' encoding required for always_fuse strategy")
                        
                        query_file_old = query_files[old_model_name]
                        print(f"  Loading OLD queries from session {past_session} for fusion: {Path(query_file_old).name}")
                        _, session_query_embs_old = read_jsonl(query_file_old)
                        cumulative_query_embs_old.extend(session_query_embs_old)
                        
                        # Validate query IDs against original data if available
                        if data_files and stream_num in data_files['queries'] and past_session in data_files['queries'][stream_num]:
                            original_query_file = data_files['queries'][stream_num][past_session]
                            print(f"  [VALIDATION] Checking query IDs against original data...")
                            query_validation = validate_ids_against_original(
                                loaded_ids=session_query_ids,
                                original_file=original_query_file,
                                file_type='queries',
                                stream=stream_num,
                                session=past_session,
                                model_used='new+old'
                            )
                            all_validation_results.append(query_validation)
                            
                            if query_validation['perfect_match']:
                                print(f"  ✓ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} (100% match, fusion enabled)")
                            else:
                                print(f"  ✗ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} "
                                      f"({query_validation['match_percentage']:.2f}% match, fusion enabled)")
                                if query_validation['missing_ids_count'] > 0:
                                    print(f"    Missing {query_validation['missing_ids_count']} IDs")
                                if query_validation['extra_ids_count'] > 0:
                                    print(f"    Extra {query_validation['extra_ids_count']} IDs")
                        
                    elif query_strategy == 'switch':
                        if query_adapter:
                            if adapter_before_switch:
                                # baseline14: Use adapter BEFORE switch, then new model AFTER switch
                                # Once session_num >= switch_session, ALL past queries use new model
                                if past_session < baseline_config.get('switch_session', 2):
                                    session_query_model = query_adapter
                                    session_query_model_name = f"{old_model_name} (refined: {query_adapter})"
                                else:
                                    session_query_model = new_model_name
                                    session_query_model_name = new_model_name
                            else:
                                # baseline11/baseline19: Use old model BEFORE switch, then adapter AFTER switch
                                # Use past_session to decide per-past-session model (not session_num)
                                # Q0, Q1 before switch always loaded with old; Q2+ with adapter
                                if past_session >= baseline_config.get('switch_session', 2):
                                    session_query_model = query_adapter
                                    session_query_model_name = f"{old_model_name} (refined: {query_adapter})"
                                else:
                                    session_query_model = old_model_name
                                    session_query_model_name = old_model_name
                        else:
                            # No adapter: once session_num >= switch_session, ALL past queries use new model
                            if session_num >= baseline_config.get('switch_session', 2):
                                session_query_model = new_model_name
                                session_query_model_name = new_model_name
                            else:
                                session_query_model = old_model_name
                                session_query_model_name = old_model_name
                    else:
                        session_query_model = current_query_model
                        session_query_model_name = current_query_model_name
                    
                    # For strategies that don't handle their own loading (exclude strategies that manage loading themselves)
                    if query_strategy not in ['always_fuse', 'fuse_before_switch', 'fuse_after_switch', 'always_rep_fuse', 'rep_fuse_before_switch', 'rep_fuse_after_switch']:
                        # Check if the required query model exists in metadata
                        if session_query_model not in query_files:
                            print(f"  ✗ ERROR: Query model '{session_query_model}' not found in metadata for stream {stream_num}, session {past_session}")
                            print(f"    Available query models: {list(query_files.keys())}")
                            raise KeyError(f"Query model '{session_query_model}' not found in metadata")
                        
                        query_file = query_files[session_query_model]
                        
                        print(f"  Loading queries from session {past_session} with {session_query_model_name}: {Path(query_file).name}")
                        session_query_ids, session_query_embs = read_jsonl(query_file)
                        
                        cumulative_query_ids.extend(session_query_ids)
                        cumulative_query_embs.extend(session_query_embs)
                        
                        # Validate query IDs against original data if available
                        if data_files and stream_num in data_files['queries'] and past_session in data_files['queries'][stream_num]:
                            original_query_file = data_files['queries'][stream_num][past_session]
                            print(f"  [VALIDATION] Checking query IDs against original data...")
                            query_validation = validate_ids_against_original(
                                loaded_ids=session_query_ids,
                                original_file=original_query_file,
                                file_type='queries',
                                stream=stream_num,
                                session=past_session,
                                model_used=session_query_model
                            )
                            all_validation_results.append(query_validation)
                            
                            if query_validation['perfect_match']:
                                print(f"  ✓ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} (100% match, model={session_query_model_name})")
                            else:
                                print(f"  ✗ Query IDs: {query_validation['loaded_count']}/{query_validation['original_count']} "
                                      f"({query_validation['match_percentage']:.2f}% match, model={session_query_model_name})")
                                if query_validation['missing_ids_count'] > 0:
                                    print(f"    Missing {query_validation['missing_ids_count']} IDs")
                                if query_validation['extra_ids_count'] > 0:
                                    print(f"    Extra {query_validation['extra_ids_count']} IDs")
                        else:
                            print(f"  [VALIDATION] Skipping query ID validation (original data not available)")

                        # Load old model queries if rank fusion is enabled (for strategies that need it)
                        # always_fuse, fuse_before_switch, and fuse_after_switch handle their own loading
                        if use_rank_fusion and query_strategy not in ['always_fuse', 'fuse_before_switch', 'fuse_after_switch']:
                            query_file_old = query_files[old_model_name]
                            print(f"  Loading {old_model_name} queries from session {past_session} for fusion: {Path(query_file_old).name}")
                            _, session_query_embs_old = read_jsonl(query_file_old)
                            cumulative_query_embs_old.extend(session_query_embs_old)
            
                # Track which queries were introduced in which session
                if past_session not in stream_queries_by_session:
                    stream_queries_by_session[past_session] = session_query_ids
            
            # ========================================
            # PART 5: Evaluate cumulative queries on index(es)
            # ========================================
            if cumulative_query_ids:
                # Dynamic fusion control for conditional strategies
                effective_use_fusion = use_rank_fusion
                if query_strategy == 'fuse_before_switch' and session_num >= baseline_config.get('switch_session', 2):
                    effective_use_fusion = False
                    print(f"\n  Evaluating {len(cumulative_query_ids)} cumulative queries with {current_query_model_name} (fusion disabled after switch)...")
                elif query_strategy == 'fuse_after_switch' and session_num < baseline_config.get('switch_session', 2):
                    effective_use_fusion = False
                    print(f"\n  Evaluating {len(cumulative_query_ids)} cumulative queries with {current_query_model_name} (fusion disabled before switch)...")
                elif query_strategy == 'rep_fuse_before_switch':
                    # Rep fusion is applied to embeddings directly, not via rank fusion
                    effective_use_fusion = False
                    if session_num < baseline_config.get('switch_session', 2):
                        print(f"\n  Evaluating {len(cumulative_query_ids)} cumulative queries with {current_query_model_name} (rep fusion before switch)...")
                    else:
                        print(f"\n  Evaluating {len(cumulative_query_ids)} cumulative queries with {current_query_model_name} (new only after switch)...")
                elif query_strategy == 'rep_fuse_after_switch':
                    # Rep fusion is applied to embeddings directly, not via rank fusion
                    effective_use_fusion = False
                    if session_num < baseline_config.get('switch_session', 2):
                        print(f"\n  Evaluating {len(cumulative_query_ids)} cumulative queries with {current_query_model_name} (old only before switch)...")
                    else:
                        print(f"\n  Evaluating {len(cumulative_query_ids)} cumulative queries with {current_query_model_name} (rep fusion after switch)...")
                else:
                    print(f"\n  Evaluating {len(cumulative_query_ids)} cumulative queries (all with {current_query_model_name})...")
                
                if effective_use_fusion:
                    print(f"  Rank fusion enabled - combining {old_model_name} and {new_model_name} query results")
                if use_multiple_indexes:
                    print(f"  Using {len(session_indexes)} separate indexes with merge method: {merge_method}")
                
                # Query embeddings are already decoded from JSONL - no need to convert IDs
                # (only documents need numeric IDs for Seismic indexing)
                query_embs_decoded = cumulative_query_embs
                
                # Apply representation fusion for rep fusion strategies
                if query_strategy == 'always_rep_fuse' and cumulative_query_embs_old:
                    print(f"  Applying representation fusion to {len(cumulative_query_embs)} queries...")
                    query_embs_decoded = representation_fusion(
                        old_reps_decoded=cumulative_query_embs_old,
                        new_reps_decoded=cumulative_query_embs,
                        weight_old=0.5,
                        weight_new=0.5
                    )
                    print(f"  Representation fusion complete: merged {len(query_embs_decoded)} query embeddings")
                    query_embs_decoded_old = None  # No rank fusion needed
                elif query_strategy == 'rep_fuse_before_switch':
                    # Apply rep fusion only BEFORE switch
                    if session_num < baseline_config.get('switch_session', 2) and cumulative_query_embs_old:
                        print(f"  Applying representation fusion to {len(cumulative_query_embs)} queries (before switch)...")
                        query_embs_decoded = representation_fusion(
                            old_reps_decoded=cumulative_query_embs_old,
                            new_reps_decoded=cumulative_query_embs,
                            weight_old=0.5,
                            weight_new=0.5
                        )
                        print(f"  Representation fusion complete: merged {len(query_embs_decoded)} query embeddings")
                        query_embs_decoded_old = None  # No rank fusion needed
                    else:
                        # After switch: use new queries only, no fusion
                        query_embs_decoded_old = None
                elif query_strategy == 'rep_fuse_after_switch':
                    # Apply rep fusion only AFTER switch
                    if session_num >= baseline_config.get('switch_session', 2) and cumulative_query_embs_old:
                        print(f"  Applying representation fusion to {len(cumulative_query_embs)} queries (after switch)...")
                        query_embs_decoded = representation_fusion(
                            old_reps_decoded=cumulative_query_embs_old,
                            new_reps_decoded=cumulative_query_embs,
                            weight_old=0.5,
                            weight_new=0.5
                        )
                        print(f"  Representation fusion complete: merged {len(query_embs_decoded)} query embeddings")
                        query_embs_decoded_old = None  # No rank fusion needed
                    else:
                        # Before switch: use old queries only, no fusion
                        query_embs_decoded_old = None
                else:
                    # Use old query embeddings directly if using rank fusion
                    query_embs_decoded_old = None
                    if use_rank_fusion and cumulative_query_embs_old:
                        query_embs_decoded_old = cumulative_query_embs_old
                
                # Perform search based on index strategy
                if use_multiple_indexes:
                    # Use multiple indexes approach
                    results, search_time = perform_search_multiple_indexes(
                        query_embeddings_decoded=query_embs_decoded,
                        corpus_indexes=session_indexes,
                        top_k=args.top_k,
                        merge_method=merge_method
                    )
                else:
                    # Use single cumulative index
                    results, search_time = perform_search(
                        query_embeddings_decoded=query_embs_decoded,
                        query_embeddings_decoded_old=query_embs_decoded_old,
                        top_k=args.top_k,
                        corpus_index=index,
                        use_rank_fusion=effective_use_fusion,
                        save_rank_fusion_splits=effective_use_fusion,
                        rank_fusion_output_dir=session_dir,
                    )
                
                # Convert results back to string IDs
                results = convert_results_to_string_ids(results, cumulative_doc_ids)
                
                # Write TREC results
                trec_file = session_dir / "results.trec"
                write_trec_results(trec_file, results, cumulative_query_ids)
                
                # Load qrels only up to current session (no future leakage!)
                print(f"  Loading qrels from session 0 to {session_num} (temporal constraint)...")
                cumulative_qrels = []
                for past_session in range(session_num + 1):
                    if past_session in encoded_data['qrels'].get(stream_num, {}):
                        qrel_file = encoded_data['qrels'][stream_num][past_session]
                        print(f"    Session {past_session}: {Path(qrel_file).name}")
                        qrels = load_qrels_from_file(qrel_file)
                        cumulative_qrels.append(qrels)
                
                merged_qrels = merge_qrels(cumulative_qrels)
                print(f"  Total qrels loaded: {len(merged_qrels)} queries")
                
                # Evaluate
                metrics = evaluate_results(results, merged_qrels, cumulative_query_ids)
                success_per_query = compute_success_at_k(results, merged_qrels, cumulative_query_ids, k=5)
                
                # Write metrics
                metrics_file = session_dir / "metrics.json"
                write_evaluation_metrics(metrics_file, metrics)
                
                # Store results with file paths for tracking
                stream_results_per_session[session_num] = {
                    'metrics': metrics,
                    'success_per_query': success_per_query,
                    'num_docs': total_docs_written,
                    'num_queries': len(cumulative_query_ids),
                    'query_model_used': current_query_model,  # Track which model was used
                    'query_encoder': current_query_model_name,  # Actual model name
                    'doc_encoder': doc_model_used_name,  # Actual model name
                    'doc_embedding_file': str(doc_file),  # Path to document embeddings used
                    'query_embedding_file': str(query_files[current_query_model]) if 'query_files' in locals() else 'N/A',  # Path to query embeddings used
                }
                
                # Clean up
                del results, query_embs_decoded
                if query_embs_decoded_old is not None:
                    del query_embs_decoded_old
                gc.collect()
            
            # Delete the index(es) after evaluation to free memory
            if use_multiple_indexes:
                # Keep session indexes for next iteration
                pass
            else:
                # Delete single cumulative index
                del index
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # No longer saving ID mapping - using ephemeral positional mapping
        
        # Clean up session indexes and temp files after stream is complete
        if use_multiple_indexes and session_indexes:
            print(f"\n  Cleaning up {len(session_indexes)} session indexes...")
            for idx in session_indexes:
                del idx
            session_indexes.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # ========================================
        # STREAM CLEANUP: Remove temporary session JSONL files (if any remain)
        # ========================================
        print(f"\n[Stream {stream_num} Cleanup] Checking for remaining temporary files...")
        files_deleted = 0
        # Remove temporary session JSONL files with numeric IDs (in case any weren't deleted)
        for temp_file in cumulative_doc_jsonl_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
                print(f"  Deleted remaining temp JSONL: {Path(temp_file).name}")
                files_deleted += 1
        
        if files_deleted == 0:
            print(f"  All temporary files already cleaned up ✓")
        
        # Compute success metrics for this stream
        print(f"\n[Success@5] Computing session-level Success@5 metrics for stream {stream_num}...")
        stream_success_metrics = compute_session_success_metrics(stream_results_per_session, stream_queries_by_session)
        
        # Write stream summary
        stream_summary_file = stream_dir / "summary.json"
        with open(stream_summary_file, 'w') as f:
            json.dump({
                'stream': stream_num,
                'baseline': baseline_config,
                'old_model': old_model_name,
                'new_model': new_model_name,
                'results_per_session': {
                    str(k): {
                        'metrics': {str(mk): mv for mk, mv in v['metrics'].items()},
                        'num_docs': v['num_docs'],
                        'num_queries': v['num_queries'],
                        'query_encoder': v['query_encoder'],
                        'doc_encoder': v['doc_encoder'],
                        'doc_embedding_file': v.get('doc_embedding_file', 'N/A'),
                        'query_embedding_file': v.get('query_embedding_file', 'N/A'),
                    } 
                    for k, v in stream_results_per_session.items()
                },
                'success_at_5': {
                    'per_query_session': {f'Q{i}_at_S{s}': v for (i, s), v in stream_success_metrics['per_query_session_success'].items()},
                    'macro_per_session': {f'S{s}': v for s, v in stream_success_metrics['macro_success_per_session'].items()},
                    'relative_gains': stream_success_metrics['relative_gains'],
                    'mean_gain': stream_success_metrics['mean_gain'],
                    'std_gain': stream_success_metrics['std_gain'],
                }
            }, f, indent=4)
        
        # Print validation summary for this stream
        stream_validations = [v for v in all_validation_results if v['stream'] == stream_num]
        if stream_validations:
            print_validation_summary(stream_validations, f"{baseline_name} - Stream {stream_num}")
        
        print(f"\n{'='*80}")
        print(f"Stream {stream_num} completed. Summary written to {stream_summary_file}")
        print(f"{'='*80}")
        
        # Store stream results
        all_streams_results[stream_num] = {
            'results_per_session': stream_results_per_session,
            'success_metrics': stream_success_metrics,
            'queries_by_session': stream_queries_by_session
        }
        
        # Clean up stream-specific cumulative_doc_ids before next stream
        del cumulative_doc_ids
        gc.collect()
    
    # ========================================
    # WRITE OVERALL SUMMARY
    # ========================================
    overall_summary_file = output_dir / baseline_name / "overall_summary.json"
    
    # Determine which models are used for queries and documents based on baseline strategy
    if baseline_config.get('query_model') == 'old' or baseline_config.get('query_model_strategy') == 'old':
        query_encoder_model = old_model_name
    elif baseline_config.get('query_model') == 'new' or baseline_config.get('query_model_strategy') == 'new':
        query_encoder_model = new_model_name
    elif baseline_config.get('query_model') == 'old_refined_tiny':
        query_encoder_model = f"{old_model_name} (adapter: old_refined_tiny)"
    elif baseline_config.get('query_model') == 'new_refined_doc':
        query_encoder_model = f"{new_model_name} (adapter: new_refined_doc)"
    elif baseline_config.get('query_model') == 'new_refined_lexical':
        query_encoder_model = f"{new_model_name} (adapter: new_refined_lexical)"
    elif baseline_config.get('query_model') == 'switch' or baseline_config.get('query_model_strategy') == 'switch':
        query_encoder_model = f"{old_model_name} → {new_model_name} (switch at session {baseline_config.get('switch_session', 'N/A')})"
    elif baseline_config.get('query_model_strategy') == 'always_rep_fuse':
        query_encoder_model = f"{old_model_name} + {new_model_name} (always rep fused)"
    elif baseline_config.get('query_model_strategy') == 'rep_fuse_before_switch':
        query_encoder_model = f"{old_model_name} + {new_model_name} (rep fused before S{baseline_config.get('switch_session', 'N/A')}), then {new_model_name} only"
    elif baseline_config.get('query_model_strategy') == 'rep_fuse_after_switch':
        query_encoder_model = f"{old_model_name} only (before S{baseline_config.get('switch_session', 'N/A')}), then {old_model_name} + {new_model_name} (rep fused)"
    elif baseline_config.get('query_model_strategy') == 'always_fuse':
        query_encoder_model = f"{old_model_name} + {new_model_name} (always fused)"
    elif baseline_config.get('query_model_strategy') == 'fuse_before_switch':
        query_encoder_model = f"{old_model_name} + {new_model_name} (fused before S{baseline_config.get('switch_session', 'N/A')}), then {new_model_name} only"
    elif baseline_config.get('query_model_strategy') == 'fuse_after_switch':
        query_encoder_model = f"{old_model_name} only (before S{baseline_config.get('switch_session', 'N/A')}), then {old_model_name} + {new_model_name} (fused)"
    else:
        query_encoder_model = "mixed/unknown"
    
    if baseline_config.get('doc_model_strategy') == 'old':
        doc_encoder_model = old_model_name
    elif baseline_config.get('doc_model_strategy') == 'new':
        doc_encoder_model = new_model_name
    elif baseline_config.get('doc_model_strategy') == 'switch':
        doc_encoder_model = f"{old_model_name} → {new_model_name} (switch at session {baseline_config.get('switch_session', 'N/A')})"
    else:
        doc_encoder_model = "mixed/unknown"
    
    with open(overall_summary_file, 'w') as f:
        json.dump({
            'baseline': baseline_config,
            'old_model': old_model_name,
            'new_model': new_model_name,
            'query_encoder': query_encoder_model,
            'doc_encoder': doc_encoder_model,
            'streams': {
                str(stream_num): {
                    'results_per_session': {
                        str(k): {
                            'metrics': {str(mk): mv for mk, mv in v['metrics'].items()},
                            'num_docs': v['num_docs'],
                            'num_queries': v['num_queries'],
                            'query_encoder': v['query_encoder'],
                            'doc_encoder': v['doc_encoder'],
                        } 
                        for k, v in stream_data['results_per_session'].items()
                    },
                    'success_at_5': {
                        'per_query_session': {f'Q{i}_at_S{s}': v for (i, s), v in stream_data['success_metrics']['per_query_session_success'].items()},
                        'macro_per_session': {f'S{s}': v for s, v in stream_data['success_metrics']['macro_success_per_session'].items()},
                        'mean_gain': stream_data['success_metrics']['mean_gain'],
                        'std_gain': stream_data['success_metrics']['std_gain'],
                    }
                }
                for stream_num, stream_data in all_streams_results.items()
            }
        }, f, indent=4)
    
    # Save validation results
    if all_validation_results:
        validation_file = output_dir / baseline_name / "id_validation.json"
        with open(validation_file, 'w') as f:
            json.dump(all_validation_results, f, indent=4)
        print(f"\nID validation results written to {validation_file}")
        
        # Print final validation summary
        print_validation_summary(all_validation_results, baseline_name)
    
    print(f"\n{'='*80}")
    print(f"{baseline_name} completed. Overall summary written to {overall_summary_file}")
    print(f"{'='*80}")
    
    return all_streams_results

def validate_ids_against_original(loaded_ids: List[str], original_file: str, 
                                   file_type: str, stream: int, session: int, 
                                   model_used: str) -> Dict:
    """
    Validate that loaded IDs match the original data file.
    
    Args:
        loaded_ids: IDs loaded from pre-encoded file
        original_file: Path to original data file (parquet for docs, jsonl for queries)
        file_type: 'documents' or 'queries'
        stream: Stream number
        session: Session number
        model_used: Which model was used ('old' or 'new')
    
    Returns:
        Dictionary with validation results
    """
    if file_type == 'documents':
        original_ids, _ = load_documents_from_parquet(original_file)
    elif file_type == 'queries':
        original_ids, _ = load_queries_from_jsonl(original_file)
    else:
        raise ValueError(f"Unknown file_type: {file_type}")
    
    loaded_set = set(loaded_ids)
    original_set = set(original_ids)
    
    missing_ids = original_set - loaded_set
    extra_ids = loaded_set - original_set
    overlap = len(loaded_set & original_set)
    
    match_percentage = (overlap / len(original_set) * 100) if original_set else 0
    perfect_match = (missing_ids == set() and extra_ids == set())
    
    validation_result = {
        'stream': stream,
        'session': session,
        'type': file_type,
        'model_used': model_used,
        'original_count': len(original_set),
        'loaded_count': len(loaded_set),
        'overlap_count': overlap,
        'match_percentage': match_percentage,
        'perfect_match': perfect_match,
        'missing_ids_count': len(missing_ids),
        'extra_ids_count': len(extra_ids),
        'missing_ids_sample': list(missing_ids)[:10] if missing_ids else [],
        'extra_ids_sample': list(extra_ids)[:10] if extra_ids else [],
    }
    
    return validation_result

def print_validation_summary(validation_results: List[Dict], baseline_name: str):
    """
    Print a summary table of ID validation results.
    """
    print(f"\n{'='*100}")
    print(f"ID VALIDATION SUMMARY - {baseline_name}")
    print(f"{'='*100}")
    print(f"{'Type':<12} {'Stream':<8} {'Session':<8} {'Model':<8} {'Original':<10} {'Loaded':<10} {'Match%':<10} {'Perfect':<10}")
    print(f"{'-'*100}")
    
    for result in validation_results:
        status = '✓ YES' if result['perfect_match'] else '✗ NO'
        print(f"{result['type']:<12} {result['stream']:<8} {result['session']:<8} "
              f"{result['model_used']:<8} {result['original_count']:<10} "
              f"{result['loaded_count']:<10} {result['match_percentage']:<10.2f} {status:<10}")
        
        if not result['perfect_match']:
            if result['missing_ids_count'] > 0:
                print(f"  ⚠ Missing {result['missing_ids_count']} IDs. Sample: {result['missing_ids_sample'][:3]}")
            if result['extra_ids_count'] > 0:
                print(f"  ⚠ Extra {result['extra_ids_count']} IDs. Sample: {result['extra_ids_sample'][:3]}")
    
    print(f"{'='*100}\n")

def load_encoded_metadata(metadata_file: Path) -> Dict:
    """
    Load metadata.json with paths to pre-encoded files.
    """
    print(f"Loading encoded metadata from: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Convert string keys back to integers
    # Structure: {stream: {session: model: path}}
    metadata_clean = {
        'queries': {int(stream): {int(sess): v for sess, v in sessions.items()} 
                   for stream, sessions in metadata['queries'].items()},
        'documents': {int(stream): {int(sess): v for sess, v in sessions.items()} 
                     for stream, sessions in metadata['documents'].items()},
        'qrels': {int(stream): {int(sess): v for sess, v in sessions.items()} 
                 for stream, sessions in metadata['qrels'].items()}
    }
    
    return metadata_clean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming IR evaluation with LOTTE dataset.")
    parser.add_argument("--old_model", type=str, default="naver/splade-v3-tiny", help="Old model name")
    parser.add_argument("--new_model", type=str, default="naver/splade-v3", help="New model name")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing LOTTE streaming files (required if not using --skip_encoding)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100, help="Top-k results for search")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Chunk size for document encoding")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for reading JSONL files")
    parser.add_argument("--baselines", type=str, nargs='+', default=['all'], 
                       help="Which baselines to run: baseline1, baseline2, baseline3, baseline4, baseline5, or 'all'")
    parser.add_argument("--skip_encoding", action='store_true',
                       help="Skip encoding phase and use existing pre-encoded files")
    parser.add_argument("--metadata_path", type=str, default=None,
                       help="Path to metadata.json (default: output_dir/pre_encoded/metadata.json)")
    parser.add_argument("--stream", type=int, default=None, choices=[1, 2, 3],
                       help="Process only a specific stream (1-3). Use with SLURM job arrays.")
    parser.add_argument("--use_multiple_indexes", action='store_true',
                       help="Use multiple indexes (one per session) instead of single cumulative index")
    parser.add_argument("--merge_method", type=str, default='max', choices=['max', 'rrf'],
                       help="Method to merge results when using multiple indexes: 'max' or 'rrf' (default: max)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle pre-encoding
    if args.skip_encoding:
        # Load existing metadata - this contains all file paths
        metadata_file = Path(args.metadata_path) if args.metadata_path else output_dir / "pre_encoded" / "metadata.json"
        print(f"\nLoading pre-encoded metadata from: {metadata_file}")
        encoded_data = load_encoded_metadata(metadata_file)
        print(f"Loaded metadata for {len(encoded_data['documents'])} document sessions and {len(encoded_data['queries'])} query sessions")
        
        # Try to discover original data files for validation if data_dir is provided
        if args.data_dir:
            data_dir = Path(args.data_dir)
            print(f"\nDiscovering original files from {data_dir} for validation...")
            data_files = discover_lotte_files(data_dir)
        else:
            print("\nWarning: --data_dir not provided, ID validation will be skipped")
            data_files = None
        
        print(f"\nUsing pre-encoded files:")
        print(f"  Documents: {sum(len(d) for d in encoded_data['documents'].values())} stream-session combinations")
        print(f"  Queries: {sum(len(q) for q in encoded_data['queries'].values())} stream-session combinations")
        print(f"  Qrels: {sum(len(q) for q in encoded_data['qrels'].values())} files across {len(encoded_data['qrels'])} streams")
        
    else:
        # Validate that data_dir is provided when not skipping encoding
        if args.data_dir is None:
            parser.error("--data_dir is required when not using --skip_encoding")
        
        data_dir = Path(args.data_dir)
        
        # Discover LOTTE files for encoding
        print("Discovering LOTTE streaming files...")
        data_files = discover_lotte_files(data_dir)
        
        print(f"\nFound files:")
        print(f"  Documents: {sum(len(d) for d in data_files['documents'].values())} files across {len(data_files['documents'])} streams")
        print(f"  Queries: {sum(len(q) for q in data_files['queries'].values())} files across {len(data_files['queries'])} streams")
        print(f"  Qrels: {sum(len(q) for q in data_files['qrels'].values())} files across {len(data_files['qrels'])} streams")

        # Pre-encode all data
        print("\n" + "="*80)
        print("Starting pre-encoding phase")
        print("="*80)
        
        old_model = SparseEncoder(args.old_model)
        new_model = SparseEncoder(args.new_model)
        
        encoded_data = pre_encode_all_data(
            data_files=data_files,
            old_model=old_model,
            new_model=new_model,
            output_dir=output_dir,
            args=args
        )
        
        # Free up model memory
        del old_model, new_model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Define baselines (loaded from bc/configs/baselines_sparse.yaml)
    _baseline_cfg_mgr = BaselineConfigManager()
    baselines = _baseline_cfg_mgr.get_baselines(args)
    
    # Run selected baselines
    baseline_keys = args.baselines if 'all' not in args.baselines else list(baselines.keys())
    print("Running baselines:", baseline_keys)
    all_baseline_results = {}
    
    for baseline_key in baseline_keys:
        if baseline_key not in baselines:
            print(f"Warning: Unknown baseline '{baseline_key}', skipping...")
            continue
        
        stream_results = run_streaming_baseline(
            baseline_config=baselines[baseline_key],
            encoded_data=encoded_data,
            data_files=data_files,
            output_dir=output_dir,
            args=args
        )
        
        all_baseline_results[baseline_key] = stream_results
