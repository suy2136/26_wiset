#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-run}"
if [[ "$MODE" != "run" && "$MODE" != "resume" && "$MODE" != "dry-run" ]]; then
  echo "usage: $0 [run|resume|dry-run]" >&2
  exit 2
fi
if [[ -z "${SEARCH_RUN_DIR:-}" ]]; then
  echo "Set SEARCH_RUN_DIR to the completed kinematic search directory." >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19_kinematic_finalists_compact/$RUN_ID}"
mkdir -p "$(dirname "$OUTPUT_DIR")"
ARGS=(
  --search-run-dir "$SEARCH_RUN_DIR"
  --output-dir "$OUTPUT_DIR"
  --device "${DEVICE:-cuda}"
)
[[ "$MODE" == "resume" ]] && ARGS+=(--resume)
[[ "$MODE" == "dry-run" ]] && ARGS+=(--dry-run)

if [[ "$MODE" != "dry-run" ]]; then
  PYTHONPATH="$PWD" python -m unittest \
    tests.test_vp_kinematic_finalists_compact
fi

python analysis/evaluate_kinematic_selector_finalists_compact.py "${ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}_launcher.log"
