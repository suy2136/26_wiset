#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-full}"
if [[ "$MODE" != "full" && "$MODE" != "resume" && "$MODE" != "dry-run" ]]; then
  echo "Usage: bash scripts/run_abr_c_matched_server1.sh [full|resume|dry-run]"
  exit 2
fi

python -c "import torch, peft, torch_incremental_pca; print('AdaLoRA/EVA dependencies OK')"
ARGS=()
[[ "$MODE" == "dry-run" ]] && ARGS+=(--dry-run)
[[ "$MODE" == "resume" ]] && ARGS+=(--resume)
python analysis/run_abr_c_matched_server1.py "${ARGS[@]}"
