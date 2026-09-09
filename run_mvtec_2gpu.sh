#!/usr/bin/env bash
# Launch full MVTec-AD evaluation split across GPU0 and GPU1, detached via nohup+setsid.
set -u
PY=/user/pfy/anaconda3/envs/anomalyagent/bin/python
REPO=/data/pfy/AgentIAD/AnomalyAgent
QWEN=/data/pfy/MLLMs/Qwen3-VL-4B-Instruct
EMB=/data/pfy/MLLMs/Qwen3-4B

COMMON_ARGS=(
  --device cuda
  --concurrent 4
  --uncertain_score 0.5
  --max_batch_size 8
  --max_wait_ms 50
  --attn_implementation flash_attention_2
  --counterfactual_top_k 3
  --super_resolution_version nongan
  --planner_model_type local
  --reflector_model_type local
  --reasoner_model_type local
  --image_description_model_type local
  --atomic_candidate_llm_model_type local
  --api_max_concurrency 16
  --use_responses_api false
  --tool_call_threshold 5
  --raw_to_reasoner true
  --raw_to_reflector true
  --normal_tolerance 2
  --resize false
  --embedding_model_path "$EMB"
  --qwen_model_path "$QWEN"
)

cd "$REPO" || exit 1

for spec in "0:mvtec_gpu0" "1:mvtec_gpu1"; do
  gpu="${spec%%:*}"
  name="${spec##*:}"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    setsid nohup "$PY" evaluate_mvtec_v12.py \
      --data_path "/data/pfy/AgentIAD/data/$name" \
      --save_path "/data/pfy/AgentIAD/results/$name" \
      "${COMMON_ARGS[@]}" \
      > "/data/pfy/AgentIAD/results/$name/run.log" 2>&1 < /dev/null &
  echo "launched $name on GPU $gpu pid $!"
done
echo "all launched"
