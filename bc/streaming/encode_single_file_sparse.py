"""
Script to encode a single file (document or query) with multiple models.
Designed to be run as part of a SLURM job array.
Optimized for sparse embeddings - only non-zero weights are stored.
Supports both HuggingFace models and local model paths.
"""
import argparse
import json
from pathlib import Path
from sentence_transformers import SparseEncoder
import torch
import gc
import os
from search_streaming_sparse import (
    load_documents_from_parquet,
    load_queries_from_jsonl,
    encode_and_save_to_jsonl
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_model_short_name(model_path: str) -> str:
    """
    Extract a short descriptive name from the model path.
    Examples:
        'naver/splade-v3-tiny' -> 'splade-v3-tiny'
        'naver/splade-v3' -> 'splade-v3'
        '/path/to/local/model/tiny' -> 'tiny'
        '/ivi/ilps/scratch/ct/jingfens_refined_sparse_model/v3_doc' -> 'v3_doc'
    """
    # Remove organization prefix if present (for HuggingFace models)
    if '/' in model_path:
        # Check if it's a local path or HuggingFace path
        path_obj = Path(model_path)
        if path_obj.exists():
            # Local path - use the directory name
            return path_obj.name
        else:
            # HuggingFace path - use everything after the last /
            return model_path.split('/')[-1]
    return model_path

def is_local_model(model_path: str) -> bool:
    """
    Check if the model path points to a local directory.
    """
    path_obj = Path(model_path)
    return path_obj.exists() and path_obj.is_dir()

def validate_local_model(model_path: str) -> bool:
    """
    Validate that a local model directory contains necessary files.
    Required files: config.json and either model.safetensors or pytorch_model.bin
    """
    path_obj = Path(model_path)
    
    has_config = (path_obj / "config.json").exists()
    has_model = (path_obj / "model.safetensors").exists() or (path_obj / "pytorch_model.bin").exists()
    has_tokenizer = (path_obj / "tokenizer_config.json").exists() or (path_obj / "vocab.txt").exists()
    
    if not has_config:
        print(f"    WARNING: Missing config.json in {model_path}")
    if not has_model:
        print(f"    WARNING: Missing model file (model.safetensors or pytorch_model.bin) in {model_path}")
    if not has_tokenizer:
        print(f"    WARNING: Missing tokenizer files in {model_path}")
    
    return has_config and has_model

def encode_file(
    input_file: str,
    file_type: str,
    models: list,
    output_dir: Path,
    chunk_size: int = 10000,
    model_labels: dict = None,
):
    """
    Encode a single file with multiple models.
    
    Args:
        input_file: Path to input file (parquet for docs, jsonl for queries)
        file_type: 'document' or 'query'
        models: List of model names/paths to use for encoding (can be HuggingFace or local paths)
        output_dir: Directory to save encoded files
        chunk_size: Chunk size for encoding
        model_labels: Dict mapping model names to custom labels for filenames
    """
    input_path = Path(input_file)
    
    # Extract stream and session from filename (e.g., test_D1_0.parquet -> D1, 0)
    filename = input_path.stem  # Remove extension
    parts = filename.split('_')
    if len(parts) >= 3:
        stream = parts[1]  # D1, D2, D3
        session = parts[2]   # 0, 1, 2, 3, 4
        file_prefix = f"{stream}_{session}"
    else:
        raise ValueError(f"Cannot parse stream/session from filename: {input_file}")
    
    print(f"\n{'='*80}")
    print(f"Encoding {file_type}: {input_path.name}")
    print(f"Stream: {stream}, Session: {session}")
    print(f"Models: {', '.join(models)}")
    print(f"{'='*80}\n")
    
    # Load data based on file type
    if file_type == 'document':
        print(f"Loading documents from parquet...")
        ids, texts = load_documents_from_parquet(input_file)
        is_query = False
        file_type_short = 'docs'
    elif file_type == 'query':
        print(f"Loading queries from jsonl...")
        ids, texts = load_queries_from_jsonl(input_file)
        is_query = True
        file_type_short = 'queries'
    else:
        raise ValueError(f"Unknown file_type: {file_type}")
    
    print(f"Loaded {len(ids)} {file_type}s")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Encode with each model
    for model_path in models:
        # Use label if provided, otherwise auto-generate from model path
        if model_labels and model_path in model_labels:
            model_short_name = model_labels[model_path]
        else:
            model_short_name = get_model_short_name(model_path)
        
        print(f"\n--- Encoding with {model_short_name} ---")
        
        # Check if local or HuggingFace model
        if is_local_model(model_path):
            print(f"    Loading from local path: {model_path}")
            if not validate_local_model(model_path):
                print(f"    ERROR: Invalid model directory. Skipping {model_path}")
                continue
        else:
            print(f"    Loading from HuggingFace: {model_path}")
        
        # Load model
        try:
            model = SparseEncoder(model_path)
            print(f"    Successfully loaded model: {model_short_name}")
        except Exception as e:
            print(f"    ERROR: Failed to load model {model_path}: {e}")
            continue
        
        # Output file path with descriptive model name
        output_jsonl = str(output_dir / f"{file_type_short}_{file_prefix}_{model_short_name}.jsonl")
        
        # Encode and save
        encode_and_save_to_jsonl(
            model=model,
            texts=texts,
            ids=ids,
            output_jsonl=output_jsonl,
            batch_size=32,
            chunk_size=chunk_size,
            is_query=is_query,
            description=f"{file_type_short.capitalize()} {file_prefix} ({model_short_name})"
        )
        
        print(f"    Saved to: {output_jsonl}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n{'='*80}")
    print(f"Completed encoding: {input_path.name}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a single file with multiple models")
    parser.add_argument("--input_file", type=str, required=True, help="Input file path")
    parser.add_argument("--file_type", type=str, required=True, choices=['document', 'query'],
                       help="Type of file to encode")
    parser.add_argument("--models", type=str, required=True,
                       help="Models configuration as JSON string or space-separated list. "
                            "JSON format: '[{\"name\": \"model1\", \"label\": \"label1\"}, ...]' "
                            "Simple format: 'model1 model2 model3'")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for encoded files")
    parser.add_argument("--chunk_size", type=int, default=10000,
                       help="Chunk size for encoding")
    
    args = parser.parse_args()
    
    # Parse models - can be JSON or simple space-separated list
    models_to_encode = []
    model_labels = {}
    models_input = args.models.strip()
    
    if models_input.startswith('['):
        # JSON format: parse and extract model names and labels
        try:
            models_config = json.loads(models_input)
            models_to_encode = [model['name'] for model in models_config]
            # Extract labels if provided
            model_labels = {model['name']: model.get('label', get_model_short_name(model['name'])) 
                           for model in models_config}
            print(f"Parsed {len(models_to_encode)} models from JSON configuration")
            print(f"Model labels: {model_labels}")
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse models JSON: {e}")
            print(f"Input: {models_input}")
            exit(1)
    else:
        # Simple space-separated format (backward compatible)
        models_to_encode = models_input.split()
    
    encode_file(
        input_file=args.input_file,
        file_type=args.file_type,
        models=models_to_encode,
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size,
        model_labels=model_labels
    )
