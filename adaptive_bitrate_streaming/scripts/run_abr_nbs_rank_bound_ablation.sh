#!/usr/bin/env bash
set -u
set -o pipefail

cd /workspace/26_wiset || exit 1
PY="${PY:-/opt/conda/envs/abr_netllm/bin/python}"
OUTPUT="${OUTPUT:-/workspace/26_wiset/adaptive_bitrate_streaming/artifacts/results/abr_nbs_rank_bound_ablation_c1536_data4_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT"

args=(
  adaptive_bitrate_streaming/analysis/run_abr_nbs_rank_bound_ablation.py
  --training-data-seed 4
  --output-dir "$OUTPUT"
  --device cuda:0
)
[[ -f "$OUTPUT/pipeline_state.json" ]] && args+=(--resume)

set +e
PYTHONUNBUFFERED=1 "$PY" "${args[@]}" 2>&1 | tee -a "$OUTPUT/pipeline.log"
rc=${PIPESTATUS[0]}
set -e

echo "Pipeline return code: $rc"
echo "OUTPUT=$OUTPUT"
exit "$rc"
