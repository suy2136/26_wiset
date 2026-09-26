#!/usr/bin/env bash
set -u
set -o pipefail

cd /workspace/26_wiset || exit 1
PY="${PY:-/opt/conda/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/26_wiset/viewport_prediction/data/experiment_runs/netllm_vs_nbs/vp_nbs_budget_scaling_data4_fixed_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_ROOT"

overall=0
for budget in 1024 1536; do
  output="$OUTPUT_ROOT/budget_${budget}"
  mkdir -p "$output"
  args=(
    analysis/run_server1_vp_data3_pipeline.py
    --training-data-seed 4
    --target-budget "$budget"
    --nbs-only
    --output-dir "$output"
    --device cuda:0
  )
  [[ -f "$output/pipeline_state.json" ]] && args+=(--resume)
  echo "===== VP NBS data4 budget ${budget} ====="
  set +e
  PYTHONUNBUFFERED=1 "$PY" "${args[@]}" 2>&1 | tee -a "$output/pipeline.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]] || ! "$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); wanted=('train_nbs','compact_nbs','evaluate_nbs_modules'); sys.exit(any(s.get('stages',{}).get(k,{}).get('status')!='complete' for k in wanted))" "$output/pipeline_state.json"; then
    echo "VP NBS budget ${budget} failed strict budget validation; continuing with the next budget."
    overall=1
  fi
done

echo "OUTPUT_ROOT=$OUTPUT_ROOT"
exit "$overall"
