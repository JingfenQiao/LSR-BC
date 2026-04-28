"""
Generate metadata.json after parallel encoding is complete.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

def generate_metadata(encoded_dir: Path, data_dir: Path, output_file: Path):
    """
    Scan the encoded directory and generate metadata.json mapping.
    """
    metadata = {
        'queries': defaultdict(lambda: defaultdict(dict)),
        'documents': defaultdict(lambda: defaultdict(dict)),
        'qrels': defaultdict(lambda: defaultdict(dict))
    }
    
    # Scan for encoded files
    for filepath in encoded_dir.glob("*.jsonl"):
        filename = filepath.stem
        
        # Parse filename: docs_D0_0_old.jsonl or queries_D1_2_new.jsonl
        parts = filename.split('_')
        
        if len(parts) >= 4:
            file_type = parts[0]  # 'docs' or 'queries'
            session = int(parts[1][1])  # D0 -> 0
            domain = int(parts[2])
            model = parts[3]  # 'old' or 'new'
            
            if file_type == 'docs':
                metadata['documents'][session][domain][model] = str(filepath)
            elif file_type == 'queries':
                metadata['queries'][session][domain][model] = str(filepath)
    
    # Add qrels paths
    qrels_dir = data_dir / "qrels"
    for session in range(4):
        for domain in range(5):
            qrel_file = qrels_dir / f"test_D{session}_{domain}.qrels"
            if qrel_file.exists():
                metadata['qrels'][session][domain] = str(qrel_file)
    
    # Convert defaultdicts to regular dicts
    metadata_clean = {
        'queries': {int(s): dict(d) for s, d in metadata['queries'].items()},
        'documents': {int(s): dict(d) for s, d in metadata['documents'].items()},
        'qrels': {int(s): dict(d) for s, d in metadata['qrels'].items()}
    }
    
    # Save metadata
    with open(output_file, 'w') as f:
        json.dump(metadata_clean, f, indent=4)
    
    print(f"Metadata generated and saved to: {output_file}")
    print(f"  Documents: {sum(len(d) for d in metadata_clean['documents'].values())} session-domain pairs")
    print(f"  Queries: {sum(len(q) for q in metadata_clean['queries'].values())} session-domain pairs")
    print(f"  Qrels: {sum(len(q) for q in metadata_clean['qrels'].values())} session-domain pairs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata.json from encoded files")
    parser.add_argument("--encoded_dir", type=str, required=True,
                       help="Directory containing encoded files")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing original LOTTE data")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output metadata file path (default: encoded_dir/metadata.json)")
    
    args = parser.parse_args()
    
    encoded_dir = Path(args.encoded_dir)
    data_dir = Path(args.data_dir)
    output_file = Path(args.output_file) if args.output_file else encoded_dir / "metadata.json"
    
    generate_metadata(encoded_dir, data_dir, output_file)
