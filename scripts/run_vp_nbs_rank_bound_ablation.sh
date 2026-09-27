#!/usr/bin/env bash
set -uo pipefail

cd /workspace/26_wiset || exit 1

PY="${PY:-/opt/conda/bin/python}"
RUN_ID="vp_nbs_rank_bound_ablation_c512_data1_$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT:-/workspace/26_wiset/viewport_prediction/data/experiment_runs/netllm_vs_nbs/$RUN_ID}"
ARGS=(--output-dir "$OUTPUT" --device "${DEVICE:-cuda:0}")

if [[ -f "$OUTPUT/pipeline_state.json" ]]; then
  ARGS+=(--resume)
fi

mkdir -p "$OUTPUT"
PYTHONUNBUFFERED=1 "$PY" analysis/run_vp_nbs_rank_bound_ablation.py "${ARGS[@]}" \
  2>&1 | tee -a "$OUTPUT/pipeline.log"
RC=${PIPESTATUS[0]}

echo "Pipeline return code: $RC"
echo "OUTPUT=$OUTPUT"
exit "$RC"
