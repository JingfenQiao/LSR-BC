import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
print(sys.path)  # For debugging purposes
from adapter.transformation_multitask_residual_add_mask import Old2NewAdapter

import argparse
import torch
import numpy as np
import json
from transformers import AutoModel
from functools import partial
from sentence_transformers import SparseEncoder
import mteb
from mteb.model_meta import ModelMeta
from typing import Union  # Add this import



# Allowed Retrieval task names (as in MTEB)
ALLOWED_TASKS = [
    "TRECCOVID",
    "NFCorpus",
    "NQ",
    "HotpotQA",
    "FiQA2018",
    "ArguAna",
    "Touche2020",
    "QuoraRetrieval",
    "DBPedia",
    "SCIDOCS",
    "FEVER",
    "ClimateFEVER",
    "SciFact",
    "MSMARCO",
    "CQADupstackRetrieval",
]

def convert_decoded_list(decoded_reps, tokenizer):
    term_ids_list = []
    weights_list = []
    for reps in decoded_reps:
        term_ids = torch.tensor([tokenizer.convert_tokens_to_ids(term) for term, weight in reps], dtype=torch.long)
        weights = torch.tensor([weight for term, weight in reps], dtype=torch.float)
        term_ids_list.append(term_ids)
        weights_list.append(weights)
    return term_ids_list, weights_list

def filter_token_ids(sentences, reps, doc_encoder):
    term_ids_list, weights_list = [], []
    special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]'}

    if not reps.is_sparse:
        reps = reps.to_sparse_coo()
    else:
        reps = reps.coalesce()


    reps = reps.coalesce()
    indices = reps.indices()   # [2, nnz]
    values = reps.values()     # [nnz]

    row_indices = indices[0]
    col_indices = indices[1]

    for i, text in enumerate(sentences):
        mask = row_indices == i
        token_ids = col_indices[mask]
        token_weights = values[mask]

        if token_ids.numel() == 0:
            term_ids_list.append(torch.empty(0, dtype=torch.long))
            weights_list.append(torch.empty(0))
            continue

        words = text.lower().split()
        keep_ids = []
        keep_vals = []

        for tid, val in zip(token_ids.tolist(), token_weights.tolist()):
            token_str = doc_encoder.tokenizer.convert_ids_to_tokens(tid)
            if token_str in special_tokens:
                continue
            if len(token_str) == 1:
                continue

            clean = token_str.replace("##", "").lower()
            if any(w.startswith(clean) for w in words):
                keep_ids.append(tid)
                keep_vals.append(val)

        term_ids_list.append(torch.tensor(keep_ids, dtype=torch.long))
        weights_list.append(torch.tensor(keep_vals, dtype=torch.float))

    return term_ids_list, weights_list

class SparseAdapterEvaluator:
    def __init__(self, new_model: str, old_model: str, adapter: Old2NewAdapter):
        self.query_encoder = SparseEncoder(new_model)
        self.doc_encoder = SparseEncoder(old_model)
        self.adapter = adapter

    def similarity(
        self,
        query_embeddings: torch.Tensor,
        corpus_embeddings: torch.Tensor,
        q_batch_size: Union[int, None] = None,  # Changed from int | None
        device: Union[torch.device, str, None] = None,
    ) -> torch.Tensor:
        """
        Compute query_embeddings @ corpus_embeddings.T with sparse inputs.
        Optimized for the common case where num_queries << num_corpus.
        """
        assert query_embeddings.is_sparse or query_embeddings.is_sparse_csr, \
            "query_embeddings must be sparse"
        assert corpus_embeddings.is_sparse or corpus_embeddings.is_sparse_csr, \
            "corpus_embeddings must be sparse"

        if device is None:
            device = query_embeddings.device

        # Move to device
        q_sp = query_embeddings.to(device)
        c_sp = corpus_embeddings.to(device)

        # Use COO for sparse mm
        if c_sp.is_sparse_csr:
            c_sp = c_sp.to_sparse_coo()
        c_sp = c_sp.coalesce()

        Q, D = q_sp.shape
        N = c_sp.size(0)

        def _mul(q_dense_batch: torch.Tensor) -> torch.Tensor:
            prod = torch.sparse.mm(c_sp, q_dense_batch.transpose(0, 1))
            return prod.transpose(0, 1).contiguous()

        if q_batch_size is None or q_batch_size >= Q:
            q_dense = q_sp.to_dense()
            return _mul(q_dense)

        out = torch.empty((Q, N), dtype=torch.float32, device=device)
        q_sp = q_sp.coalesce() if q_sp.is_sparse else q_sp
        for start in range(0, Q, q_batch_size):
            end = min(start + q_batch_size, Q)
            q_batch_dense = q_sp[start:end].to_dense()
            out[start:end] = _mul(q_batch_dense)
        return out

    def encode(
        self, 
        sentences: list[str], 
        task_name: str,
        prompt_type: Union[str, None] = None,
        batch_size: Union[int, None] = None,
        **kwargs,
    ):
        with torch.no_grad():
            if prompt_type == "query":
                reps = self.query_encoder.encode_query(sentences)
            else:
                reps = self.doc_encoder.encode_document(sentences)
                old_term_ids, old_weights = filter_token_ids(sentences, reps, self.doc_encoder)

                # Pad to same length
                max_len = max(len(t) for t in old_term_ids) if old_term_ids else 0
                padded_term_ids = torch.zeros((len(sentences), max_len), dtype=torch.long, device=reps.device)
                padded_weights = torch.zeros((len(sentences), max_len), dtype=torch.float, device=reps.device)
                
                for i in range(len(sentences)):
                    length = len(old_term_ids[i])
                    if length > 0:
                        padded_term_ids[i, :length] = old_term_ids[i]
                        padded_weights[i, :length] = old_weights[i]
                batch_size = 32
                final_reps = []
                for start in range(0, len(sentences), batch_size or len(sentences)):
                    end = min(start + (batch_size or len(sentences)), len(sentences))
                    batch_term_ids = padded_term_ids[start:end]
                    batch_weights = padded_weights[start:end]
                    batch_reps, _, _ = self.adapter(batch_term_ids, batch_weights)
                    final_reps.append(batch_reps)

                reps = torch.cat(final_reps, dim=0)
                reps = reps.to_sparse_coo().coalesce()  # coalesces duplicate indices
        return reps
  
def main():
    parser = argparse.ArgumentParser(description="Run MTEB benchmark with SparseEncoder")

    # Model args
    parser.add_argument("--old_model", type=str, default="rasyosef/splade-tiny", help="Old model name")
    parser.add_argument("--new_model", type=str, default="naver/splade-v3", help="New model name")

    # Evaluation args
    parser.add_argument("--benchmark_name", type=str, default="MTEB(eng, v2)", help="Benchmark name to load")
    parser.add_argument("--task_type", type=str, default="Retrieval", help="Filter benchmark by task type (e.g., Retrieval)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding")
    parser.add_argument("--output_folder", type=str, default="lsr-bc-evaluation-results", help="Where to save results")
    parser.add_argument("--verbosity", type=int, default=2, help="Verbosity level for MTEB")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cuda or cpu)")

    # NEW: choose specific tasks
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALLOWED_TASKS,
        help=f"One or more tasks to run (choices: {', '.join(ALLOWED_TASKS)}). If omitted, runs all { 'Retrieval' } tasks from the selected benchmark."
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List available tasks (by name) in the selected benchmark and exit."
    )

    args = parser.parse_args()

    print(f"\n=== Running MTEB Benchmark ===")
    print(f"Old Model: {args.old_model}")
    print(f"New Model: {args.new_model}")
    print(f"Benchmark: {args.benchmark_name}")
    print(f"Task Type: {args.task_type}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Output Folder: {args.output_folder}")
    if args.tasks:
        print(f"Selected Tasks: {', '.join(args.tasks)}")
    print("=" * 50)

    print("\nLoading best model for final test...")
    adapter = Old2NewAdapter.from_pretrained("/ivi/ilps/personal/jkang1/jf/lsr-bc/adapter_checkpoints/old_sp_tiny_new_sp_v3_1000000_data_multitask_under_kl_dp2_residual_mask").to(torch.device(args.device))
    adapter.eval()

    # Initialize wrapper via ModelMeta
    qe_safe = args.new_model.replace('/', '_')
    de_safe = args.old_model.replace('/', '_')
    model_meta = ModelMeta(
        loader=partial(
            SparseAdapterEvaluator,
            new_model=args.new_model,
            old_model=args.old_model,
            adapter=adapter
        ),
        name=f"jingfen/query_{qe_safe}_doc_{de_safe}",
        languages=["eng-Latn"],
        modalities=["text"],
        revision="v1",
        release_date="2025-10-16",
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=512,
        embed_dim=30522,
        license=None,
        open_weights=True,
        public_training_code="",
        public_training_data="",
        framework=["PyTorch"],
        similarity_fn_name=None,
        use_instructions=True,
        training_datasets={},
    )
    model = model_meta.load_model()
    # model.encode.batch_size = args.batch_size

    # Load benchmark and optionally list tasks
    benchmark = mteb.get_benchmark(args.benchmark_name)

    if args.list_tasks:
        names = [t.metadata.name for t in benchmark]
        print("\nAvailable tasks in this benchmark:")
        for n in sorted(names):
            print(f"- {n}")
        return

    # Filter by task type
    tasks_by_type = [t for t in benchmark if t.metadata.type == args.task_type]

    # If user specified specific tasks, filter further by those names
    if args.tasks:
        requested = set(args.tasks)
        selected = [t for t in tasks_by_type if t.metadata.name in requested]
        # Report any missing
        found_names = {t.metadata.name for t in selected}
        missing = requested - found_names
        if missing:
            print(f"Warning: These requested tasks were not found in '{args.benchmark_name}' with type '{args.task_type}': {', '.join(sorted(missing))}")
        retrieval_tasks = selected
    else:
        retrieval_tasks = tasks_by_type

    print(f"Selected {len(retrieval_tasks)} {args.task_type} tasks.\n")

    if not retrieval_tasks:
        print("No tasks to run. Exiting.")
        return

    # Run evaluation
    evaluation = mteb.MTEB(tasks=retrieval_tasks)
    results = evaluation.run(
        model,
        encode_kwargs={"batch_size": args.batch_size},
        output_folder=args.output_folder,
        save_predictions=True,
        verbosity=args.verbosity,
    )

    # Aggregate results
    all_task_avgs = []
    for r in results:
        split_scores = []
        for split, measures in r.scores.items():
            for subset in measures:
                split_scores.append(subset["main_score"])
        if split_scores:
            task_avg = float(np.mean(split_scores))
            all_task_avgs.append(task_avg)
            print(f"{r.task_name}: {task_avg:.4f}")

    if all_task_avgs:
        overall_avg = float(np.mean(all_task_avgs))
        print(f"\nOverall average over {len(all_task_avgs)} {args.task_type} tasks: {overall_avg:.4f}")


if __name__ == "__main__":
    main()
