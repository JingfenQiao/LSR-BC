import json
import ir_datasets
import ir_measures
from ir_measures import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import argparse

def fuse(runs, weights):
    fused_run = {}
    qids = set()
    for run in runs:
        qids.update(run.keys())
    for qid in qids:
        fused_run[qid] = {}
        for run in runs:
            for doc in run[qid]['docs']:
                if doc not in fused_run[qid]:
                    score = 0
                    for temp_run, weight in zip(runs, weights):
                        if doc in temp_run[qid]['docs']:
                            min_score = temp_run[qid]['min_score']
                            max_score = temp_run[qid]['max_score']
                            denominator = max_score - min_score
                            denominator = max(denominator, 1e-9)
                            score += weight * ((temp_run[qid]['docs'][doc] - min_score) / denominator)
                        else:
                            score += 0
                    fused_run[qid][doc] = score
    return fused_run

def convert2fusion(run_dict):
    run = {}
    for qid in run_dict:
        if qid not in run:
            run[qid] = {'docs': {}, 'max_score': None, 'min_score': None}
        scores = list(run_dict[qid].values())
        # print( scores )
        run[qid]['max_score'] = max(scores)
        run[qid]['min_score'] = min(scores)
        for docid, score in run_dict[qid].items():
            run[qid]['docs'][docid] = score
    return run

def rank_fusion(top_old_indices, top_new_indices, k=3, constant=60):
    unique_docs = set(top_old_indices) | set(top_new_indices)
    fused_scores = {doc: 0 for doc in unique_docs}
    
    for rank, doc in enumerate(top_old_indices, 1):
        fused_scores[doc] += 1 / (rank + constant)
    
    for rank, doc in enumerate(top_new_indices, 1):
        fused_scores[doc] += 1 / (rank + constant)
    
    # Sort by fused score descending
    fused_ranked = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
    return fused_ranked

def read_run(run_file_path):
    with open(run_file_path, 'r') as f:
        run = json.load(f)
    return run

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--old_query_model_name', type=str, required=False, default="sp_v3_doc", help='Old query model name')
    args.add_argument('--old_doc_model_name', type=str, required=False, default="sp_v3_doc", help='Old document model name')
    args.add_argument('--new_query_model_name', type=str, required=False, default="sp_v3", help='New query model name')
    args.add_argument('--new_doc_model_name', type=str, required=False, default="sp_v3", help='New document model name')
    parsed_args = args.parse_args()
    old_query_model_name = parsed_args.old_query_model_name
    old_doc_model_name = parsed_args.old_doc_model_name
    new_query_model_name = parsed_args.new_query_model_name
    new_doc_model_name = parsed_args.new_doc_model_name

    dir = "/ivi/ilps/personal/jkang1/jf/lsr-bc/results/results_normal"
    datasets = {
        "NQ": "nq",
        "HotpotQA": "hotpotqa/test",
        "FiQA2018": "fiqa/test",
        "ArguAna": "arguana",
        "Touche2020": "webis-touche2020",
        "QuoraRetrieval": "quora/test",
        "DBPedia": "dbpedia-entity/test",
        "SCIDOCS": "scidocs",
        "FEVER": "fever/test",
        "ClimateFEVER": "climate-fever",
        "SciFact": "scifact/test",
        "MSMARCO": "msmarco/dev",
        "TRECCOVID": "trec-covid",
        "NFCorpus": "nfcorpus/test",
        "CQADupstackRetrieval": "cqadupstack",
    }

    alpha = 0.5
    rank_based_fusion = {}
    for path_data_name, beir_data_name in datasets.items():
        if path_data_name == "CQADupstackRetrieval": continue
        old_query_model_old_doc_model_path = f"{dir}/q_{old_query_model_name}_d_{old_doc_model_name}/{path_data_name}_default_predictions.json"
        new_query_model_new_doc_model_path = f"{dir}/q_{new_query_model_name}_d_{new_doc_model_name}/{path_data_name}_default_predictions.json"
        new_query_model_old_doc_model_path = f"{dir}/q_{new_query_model_name}_d_{old_doc_model_name}/{path_data_name}_default_predictions.json"

        old_query_model_old_doc_model_run_res = read_run(old_query_model_old_doc_model_path)
        new_query_model_new_doc_model_run_res = read_run(new_query_model_new_doc_model_path)
        new_query_model_old_doc_model_run_res = read_run(new_query_model_old_doc_model_path)

        fusion_run_res = fuse(
            runs=[convert2fusion(old_query_model_old_doc_model_run_res), convert2fusion(new_query_model_old_doc_model_run_res)],
            weights=[alpha, (1 - alpha)],
        )
        qrels = ir_datasets.load(f'beir/{beir_data_name}').qrels
        old_query_model_old_doc_model_metrics = ir_measures.calc_aggregate([nDCG@10], qrels, old_query_model_old_doc_model_run_res)
        new_query_model_new_doc_model_metrics = ir_measures.calc_aggregate([nDCG@10], qrels, new_query_model_new_doc_model_run_res)
        new_query_model_old_doc_model_metrics = ir_measures.calc_aggregate([nDCG@10], qrels, new_query_model_old_doc_model_run_res)
        fusion_run_res = ir_measures.calc_aggregate([nDCG@10], qrels, fusion_run_res)
        print("Dataset:", path_data_name)
        print(old_query_model_old_doc_model_metrics, "Old Query Model & Old Doc Model Results Path")
        print(new_query_model_new_doc_model_metrics, "New Query Model & New Doc Model Results Path")
        print(new_query_model_old_doc_model_metrics, "New Query Model & Old Doc Model Results Path")
        print(fusion_run_res, "Fusion Results Path")
        print("\n")
        rank_based_fusion[path_data_name] = {"old_query_model_old_doc_model": old_query_model_old_doc_model_metrics[nDCG@10],
                                            "new_query_model_new_doc_model": new_query_model_new_doc_model_metrics[nDCG@10],
                                            "new_query_model_old_doc_model": new_query_model_old_doc_model_metrics[nDCG@10],
                                            "fusion": fusion_run_res[nDCG@10]}
        
    # Data
    data = rank_based_fusion
    # Relabeled model settings
    label_map = {
        'old_query_model_old_doc_model': 'Old Query + Old Doc',
        'new_query_model_new_doc_model': 'New Query + New Doc',
        'new_query_model_old_doc_model': 'New Query + Old Doc',
        'fusion': 'Rank-Based Fusion',
    }
    datasets = list(data.keys())
    categories = list(next(iter(data.values())).keys())
    num_cats = len(categories)

    # Bar placement
    bar_width = 0.18
    gap = 0.2  # space between dataset groups
    x = np.arange(len(datasets)) * (num_cats * bar_width + gap)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16,6))

    for i, category in enumerate(categories):
        values = [data[ds][category] for ds in datasets]
        ax.bar(x + i*bar_width, values, width=bar_width, label=label_map[category])

    # Axis setup
    ax.set_ylabel('nDCG@10')
    ax.set_title(f'{{old_query_model_name}} & {{new_query_model_name}}')

    # Center dataset labels under groups
    ax.set_xticks(x + (num_cats - 1) * bar_width / 2)
    ax.set_xticklabels(datasets)

    # Grid & legend
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Model Setting', loc='upper left', bbox_to_anchor=(0, 1))

    # Add value labels on top of each bar
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 0.005,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{old_query_model_name} & {new_query_model_name} rank_based_fusion.png', dpi=300)
