old_model=(rasyosef/splade-tiny sp_tiny)
new_model=(naver/splade-v3 sp_v3)

# old_model=(naver/splade-cocondenser-selfdistil sp_co_selfdistil)
# new_model=(naver/splade-v3 sp_v3)

# old_model=(naver/splade-v3-distilbert sp_v3_distil)
# new_model=(naver/splade-v3 sp_v3)

# old_model=(naver/splade-v3-doc sp_v3_doc)
# new_model=(naver/splade-v3 sp_v3)

# old_model=(naver/splade-v3-lexical sp_v3_lexical)
# new_model=(naver/splade-v3 sp_v3)


type=(2)
fusion_type=${type[$SLURM_ARRAY_TASK_ID-1]}

CUDA_VISIBLE_DEVICES=0 python bc/evaluate_beir_rep_fusion.py \
  --old_model ${old_model[0]} \
  --new_model ${new_model[0]} \
  --fusion_type 2 \
  --batch_size 256 \
  --benchmark BEIR \
  --task_type Retrieval \
  --tasks MSMARCO \
  --output_folder fusion_results/new_${new_model[1]}_old_${old_model[1]}_fusion_type_$fusion_type
