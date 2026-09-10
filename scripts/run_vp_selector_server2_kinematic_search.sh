#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-run}"
if [[ "$MODE" == "resume" && -z "${RUN_ID:-}" && -z "${OUTPUT_DIR:-}" ]]; then
  echo "resume requires RUN_ID=<existing-id> or OUTPUT_DIR=<existing-directory>" >&2
  exit 2
fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19_kinematic_selector_search/$RUN_ID}"
mkdir -p "$(dirname "$OUTPUT_DIR")"
ARGS=(
  --output-dir "$OUTPUT_DIR"
  --validation-samples "${VALIDATION_SAMPLES:-256}"
  --interaction-candidates "${INTERACTION_CANDIDATES:-32}"
  --refinement-candidates "${REFINEMENT_CANDIDATES:-12}"
  --refinement-samples "${REFINEMENT_SAMPLES:-512}"
  --full-validation-candidates "${FULL_VALIDATION_CANDIDATES:-5}"
)
if [[ "$MODE" == "dry-run" ]]; then
  ARGS+=(--dry-run)
elif [[ "$MODE" == "resume" ]]; then
  ARGS+=(--resume)
elif [[ "$MODE" != "run" ]]; then
  echo "usage: $0 [run|resume|dry-run]" >&2
  exit 2
fi
if [[ "$MODE" != "dry-run" ]]; then
  PYTHONPATH="$PWD" python -m unittest \
    tests.test_kinematic_patch_selector \
    tests.test_kinematic_patch_selector_search
fi

python analysis/search_kinematic_patch_selector.py "${ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}_launcher.log"
