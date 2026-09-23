#!/usr/bin/env bash
set -u
set -o pipefail

cd /workspace/26_wiset || exit 1
PY="${PY:-/opt/conda/envs/abr_netllm/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/26_wiset/adaptive_bitrate_streaming/artifacts/results/abr_lora_budget_scaling_data4_20260923}"
mkdir -p "$OUTPUT_ROOT"

overall=0
for budget in 512 1024; do
  output="$OUTPUT_ROOT/budget_${budget}"
  mkdir -p "$output"
  args=(
    adaptive_bitrate_streaming/analysis/run_server2_abr_data2_pipeline.py
    --training-data-seed 4
    --target-budget "$budget"
    --lora-only
    --output-dir "$output"
    --device cuda:0
  )
  [[ -f "$output/pipeline_state.json" ]] && args+=(--resume)
  echo "===== ABR data4 budget ${budget} ====="
  set +e
  PYTHONUNBUFFERED=1 "$PY" "${args[@]}" 2>&1 | tee -a "$output/pipeline.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]] || ! "$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(any(v.get('status')=='failed' for v in s.get('stages',{}).values()))" "$output/pipeline_state.json"; then
    echo "ABR budget ${budget} has failed stages; continuing with the next independent budget."
    overall=1
  fi
done

echo "OUTPUT_ROOT=$OUTPUT_ROOT"
exit "$overall"
