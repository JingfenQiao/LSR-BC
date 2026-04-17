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

class SparseEvaluator:
    def __init__(self, query_encoder=None, doc_encoder=None):
        self.query_encoder = SparseEncoder(query_encoder)
        self.doc_encoder = SparseEncoder(doc_encoder)

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
        if prompt_type == "query":
            reps = self.query_encoder.encode(sentences)
        else:
            reps = self.doc_encoder.encode_document(sentences)
        return reps

def main():
    parser = argparse.ArgumentParser(description="Run MTEB benchmark with SparseEncoder")

    # Model args
    parser.add_argument("--query_encoder", type=str, default="naver/splade-v3", help="Query encoder model name")
    parser.add_argument("--doc_encoder", type=str, default="naver/splade-v3", help="Document encoder model name")

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
    print(f"Query Encoder: {args.query_encoder}")
    print(f"Document Encoder: {args.doc_encoder}")
    print(f"Benchmark: {args.benchmark_name}")
    print(f"Task Type: {args.task_type}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Output Folder: {args.output_folder}")
    if args.tasks:
        print(f"Selected Tasks: {', '.join(args.tasks)}")
    print("=" * 50)

    # Initialize wrapper via ModelMeta
    qe_safe = args.query_encoder.replace('/', '_')
    de_safe = args.doc_encoder.replace('/', '_')
    model_meta = ModelMeta(
        loader=partial(
            SparseEvaluator,
            query_encoder=args.query_encoder,
            doc_encoder=args.doc_encoder
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
