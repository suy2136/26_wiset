#!/usr/bin/env bash
set -u
set -o pipefail

cd /workspace/26_wiset || exit 1
PY="${PY:-/opt/conda/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/26_wiset/viewport_prediction/data/experiment_runs/netllm_vs_nbs/vp_lora_budget_scaling_data4_20260923}"
mkdir -p "$OUTPUT_ROOT"

overall=0
for budget in 1024 1536; do
  output="$OUTPUT_ROOT/budget_${budget}"
  mkdir -p "$output"
  args=(
    analysis/run_server1_vp_data3_pipeline.py
    --training-data-seed 4
    --target-budget "$budget"
    --lora-only
    --output-dir "$output"
    --device cuda:0
  )
  [[ -f "$output/pipeline_state.json" ]] && args+=(--resume)
  echo "===== VP data4 budget ${budget} ====="
  set +e
  PYTHONUNBUFFERED=1 "$PY" "${args[@]}" 2>&1 | tee -a "$output/pipeline.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]] || ! "$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(any(v.get('status')=='failed' for v in s.get('stages',{}).values()))" "$output/pipeline_state.json"; then
    echo "VP budget ${budget} has failed stages; continuing with the next independent budget."
    overall=1
  fi
done

echo "OUTPUT_ROOT=$OUTPUT_ROOT"
exit "$overall"
