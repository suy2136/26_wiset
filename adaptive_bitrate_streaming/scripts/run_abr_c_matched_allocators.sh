#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-full}"
if [[ "$MODE" != "full" && "$MODE" != "dry-run" ]]; then
  echo "Usage: bash scripts/run_abr_c_matched_allocators.sh [full|dry-run]"
  exit 2
fi

python -c "import torch, peft; print('AdaLoRA/Shapley dependencies OK')"
python -c "import torch_incremental_pca; print('EVA dependency OK')"

ARGS=()
if [[ "$MODE" == "dry-run" ]]; then
  ARGS+=(--dry-run)
fi

python analysis/run_abr_c_matched_allocators.py "${ARGS[@]}"
