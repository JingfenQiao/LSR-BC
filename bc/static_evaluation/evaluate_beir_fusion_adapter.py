import argparse
import torch
import numpy as np
from functools import partial
from sentence_transformers import SparseEncoder
import mteb
from mteb.model_meta import ModelMeta
from typing import Union, List

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
    def __init__(self, 
                 old_model="naver/splade-v3", 
                 new_model="naver/splade-v3", 
                 adapter="",
                 fusion_type=1, 
                 device="cpu"):
        
        self.device = torch.device(device)
        self.fusion_type = fusion_type

        self.query_encoder_old = SparseEncoder(old_model)
        self.doc_encoder_old = SparseEncoder(old_model)

        self.query_encoder_new = SparseEncoder(new_model)
        # self.doc_encoder_new = SparseEncoder(new_model)

        if adapter:
            self.query_adapter = SparseEncoder(adapter)
            # self.doc_adapter = SparseEncoder(adapter)

    def similarity(
        self,
        query_embeddings: torch.Tensor,
        corpus_embeddings: torch.Tensor,
        q_batch_size: Union[int, None] = None,
        device: Union[torch.device, str, None] = None,
    ) -> torch.Tensor:
        """
        Compute query_embeddings @ corpus_embeddings.T with sparse inputs.
        Optimized for the common case where num_queries << num_corpus.
        """
        assert query_embeddings.is_sparse or query_embeddings.is_sparse_csr, "query_embeddings must be sparse"
        assert corpus_embeddings.is_sparse or corpus_embeddings.is_sparse_csr, "corpus_embeddings must be sparse"

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
    
    @staticmethod
    def convert_to_coo_tensor(batch_reps: List[List[tuple]], tokenizer):
        batch_size = len(batch_reps)
        all_batch_indices = []
        all_term_indices = []
        all_values = []
        
        for i, reps in enumerate(batch_reps):
            token_strings = [token for token, weight in reps]
            token_ids = tokenizer.convert_tokens_to_ids(token_strings)
            
            values = torch.tensor([weight for token, weight in reps])
            
            batch_indices = [i] * len(token_ids)
            all_batch_indices.extend(batch_indices)
            all_term_indices.extend(token_ids)
            all_values.append(values)
        
        if not all_batch_indices:
            indices = torch.empty((2, 0), dtype=torch.long)
            values = torch.empty((0,), dtype=torch.float32)
        else:
            indices = torch.tensor([all_batch_indices, all_term_indices])
            values = torch.cat(all_values)
        
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(batch_size, tokenizer.vocab_size))
        
        return sparse_tensor

    @staticmethod
    def fusion1(old_reps_decoded, new_reps_decoded):
        all_reps = []
        for old_sparse, new_sparse in zip(old_reps_decoded, new_reps_decoded):
            # Convert to dicts for easier fusion {token: weight}
            old_dict = {token: weight for token, weight in old_sparse}
            new_dict = {token: weight for token, weight in new_sparse}

            # Fusion strategy: using weighting from new model but terms from old model
            reps = []
            for token in sorted(old_dict.keys()):
                if token in new_dict:
                    reps.append((token, new_dict[token]))
                else:
                    reps.append((token, old_dict[token]))
            all_reps.append(reps)
        return all_reps

    @staticmethod
    def fusion2(old_reps_decoded, new_reps_decoded):
        all_reps = []
        for old_sparse, new_sparse in zip(old_reps_decoded, new_reps_decoded):
            # Convert to dicts for easier fusion {token: weight}
            old_dict = {token: weight for token, weight in old_sparse}
            new_dict = {token: weight for token, weight in new_sparse}

            union_tokens = set(old_dict.keys()) | set(new_dict.keys())
            reps = [(token, (old_dict.get(token, 0) + new_dict.get(token, 0)) / 2) for token in sorted(union_tokens) if (old_dict.get(token, 0) + new_dict.get(token, 0)) / 2 > 0]

            all_reps.append(reps)
        return all_reps

    @staticmethod
    def fusion3(old_reps_decoded, new_reps_decoded):
        all_reps = []
        for old_sparse, new_sparse in zip(old_reps_decoded, new_reps_decoded):
            # Convert to dicts for easier fusion {token: weight}
            old_dict = {token: weight for token, weight in old_sparse}
            new_dict = {token: weight for token, weight in new_sparse}

            # Max Fusion (Strongest Signal): For each term: Take the maximum weight from either model (0 if absent).
            union_tokens = set(old_dict.keys()) | set(new_dict.keys())
            reps = [(token, max(old_dict.get(token, 0), new_dict.get(token, 0))) for token in sorted(union_tokens) if max(old_dict.get(token, 0), new_dict.get(token, 0)) > 0]

            all_reps.append(reps)
        return all_reps

    @staticmethod
    def fusion4(old_reps_decoded, new_reps_decoded):
        all_reps = []
        for old_sparse, new_sparse in zip(old_reps_decoded, new_reps_decoded):
            # Convert to dicts for easier fusion {token: weight}
            old_dict = {token: weight for token, weight in old_sparse}
            new_dict = {token: weight for token, weight in new_sparse}

            # Term Expansion Fusion (High Precision): Only keep terms that appear in both representations, combining their weights (e.g., averaging or summing).
            intersection_tokens = set(old_dict.keys()) & set(new_dict.keys())
            reps = [(token, (old_dict[token] + new_dict[token]) / 2) for token in sorted(intersection_tokens) if (old_dict[token] + new_dict[token]) / 2 > 0]

            all_reps.append(reps)
        return all_reps

    @staticmethod
    def fusion5(old_reps_decoded, new_reps_decoded, adapter_reps_decoded):
        all_reps = []
        for old_sparse, new_sparse, adapter_sparse in zip(old_reps_decoded, new_reps_decoded, adapter_reps_decoded):
            old_dict = {token: weight for token, weight in old_sparse}
            new_dict = {token: weight for token, weight in new_sparse}
            adapter_dict = {token: weight for token, weight in adapter_sparse}

            union_tokens = set(old_dict.keys()) | set(new_dict.keys()) | set(adapter_dict.keys())
            reps = [(token, (old_dict.get(token, 0) + new_dict.get(token, 0) + adapter_dict.get(token, 0)) / 3) for token in sorted(union_tokens) if (old_dict.get(token, 0) + new_dict.get(token, 0) + adapter_dict.get(token, 0)) / 3 > 0]
            all_reps.append(reps)
        return all_reps

    @staticmethod
    def fusion6(old_reps_decoded, new_reps_decoded):
        all_reps = []
        for old_sparse, new_sparse in zip(old_reps_decoded, new_reps_decoded):
            old_dict = {token: weight for token, weight in old_sparse}
            new_dict = {token: weight for token, weight in new_sparse}

            union_tokens = set(old_dict) | set(new_dict)
            reps = []

            for token in sorted(union_tokens):
                if token in old_dict and token in new_dict:
                    weight = (old_dict[token] + new_dict[token]) / 2
                elif token in new_dict:
                    weight = new_dict[token]
                else:
                    weight = old_dict[token]

                if weight > 0:
                    reps.append((token, weight))

            all_reps.append(reps)
        return all_reps


    @staticmethod
    def fusion7(old_reps_decoded, new_reps_decoded, adapter_reps_decoded):
        all_reps = []
        for old_sparse, new_sparse, adapter_sparse in zip(old_reps_decoded, new_reps_decoded, adapter_reps_decoded):
            old_dict = {token: weight for token, weight in old_sparse}
            new_dict = {token: weight for token, weight in new_sparse}
            adapter_dict = {token: weight for token, weight in adapter_sparse}

            union_tokens = set(old_dict) | set(new_dict) | set(adapter_dict)
            reps = []

            for token in sorted(union_tokens):
                if token in old_dict and token in new_dict and token in adapter_dict:
                    weight = (old_dict[token] + new_dict[token] + adapter_dict[token]) / 3
                elif token in old_dict and token in new_dict:
                    weight = (old_dict[token] + new_dict[token]) / 2
                elif token in old_dict and token in adapter_dict:
                    weight = (old_dict[token] + adapter_dict[token]) / 2
                elif token in new_dict and token in adapter_dict:
                    weight = (new_dict[token] + adapter_dict[token]) / 2
                elif token in adapter_dict:
                    weight = adapter_dict[token]
                elif token in new_dict:
                    weight = new_dict[token]
                else:
                    weight = old_dict[token]

                if weight > 0:
                    reps.append((token, weight))

            all_reps.append(reps)
        return all_reps


    def encode(
        self, 
        sentences: list[str], 
        task_name: str,
        prompt_type: Union[str, None] = None,
        batch_size: Union[int, None] = None,
        **kwargs,
    ):
        if prompt_type == "query":
            new_reps = self.query_encoder_new.encode_query(sentences)
            new_reps_decoded = self.query_encoder_new.decode(new_reps)

            old_reps = self.query_encoder_old.encode_query(sentences)
            old_reps_decoded = self.query_encoder_old.decode(old_reps)

            fusion_func = getattr(self, f'fusion{self.fusion_type}')

            if self.fusion_type in [5, 7]:
                adapter_reps = self.query_adapter.encode_query(sentences)
                adapter_reps_decoded = self.query_adapter.decode(adapter_reps)
                fused_rep = fusion_func(old_reps_decoded, new_reps_decoded, adapter_reps_decoded)
            else:
                fused_rep = fusion_func(old_reps_decoded, new_reps_decoded)

            reps = self.convert_to_coo_tensor(fused_rep, self.query_encoder_new.tokenizer)
        else:
            reps = self.doc_encoder_old.encode_document(sentences)
        return reps

def main():
    parser = argparse.ArgumentParser(description="Run MTEB benchmark with SparseEncoder")

    # Model args
    parser.add_argument("--old_model", type=str, default="naver/splade-v3", help="Query encoder model name")
    parser.add_argument("--new_model", type=str, default="naver/splade-v3", help="Document encoder model name")
    parser.add_argument("--adapter_model", type=str, default="", help="Adapter model name")
    parser.add_argument("--fusion_type", type=int, default=1, choices=[1,2,3,4,5,6,7], help="Fusion strategy type")

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
    print(f"Query Encoder: {args.old_model}")
    print(f"Document Encoder: {args.new_model}")
    print(f"Fusion Type: {args.fusion_type}")
    print(f"Benchmark: {args.benchmark_name}")
    print(f"Task Type: {args.task_type}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Output Folder: {args.output_folder}")
    if args.tasks:
        print(f"Selected Tasks: {', '.join(args.tasks)}")
    print("=" * 50)

    # Initialize wrapper via ModelMeta
    qe_safe = args.old_model.replace('/', '_')
    de_safe = args.new_model.replace('/', '_')
    model_meta = ModelMeta(
        loader=partial(
            SparseEvaluator,
            old_model=args.old_model,
            new_model=args.new_model,
            adapter=args.adapter_model,
            fusion_type=args.fusion_type,
            device=args.device
        ),
        name=f"jingfen/old_{qe_safe}_new_{de_safe}_fusion{args.fusion_type}",
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
                main_score = subset.get('ndcg_at_10', subset.get('main_score'))
                if main_score is not None:
                    split_scores.append(main_score)
        if split_scores:
            task_avg = float(np.mean(split_scores))
            all_task_avgs.append(task_avg)
            print(f"{r.task_name}: {task_avg:.4f}")

    if all_task_avgs:
        overall_avg = float(np.mean(all_task_avgs))
        print(f"\nOverall average over {len(all_task_avgs)} {args.task_type} tasks: {overall_avg:.4f}")


if __name__ == "__main__":
    main()