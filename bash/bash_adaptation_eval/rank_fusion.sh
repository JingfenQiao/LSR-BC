old_models=("sp_tiny" "sp_v3_doc" "sp_v3_lexical" "sp_v3_distil")
new_models=("sp_v3"   "sp_v3"     "sp_v3"        "sp_v3")

idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
old_model="${old_models[$idx]}"
new_model="${new_models[$idx]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: old_model=${old_model}, new_model=${new_model}"

python bc/rank_based_fusion.py \
    --old_query_model_name "${old_model}" \
    --old_doc_model_name   "${old_model}" \
    --new_query_model_name "${new_model}" \
    --new_doc_model_name   "${new_model}"