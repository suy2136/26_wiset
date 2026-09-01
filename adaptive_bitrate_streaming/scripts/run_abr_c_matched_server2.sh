#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-full}"
if [[ "$MODE" != "full" && "$MODE" != "dry-run" ]]; then
  echo "Usage: bash scripts/run_abr_c_matched_server2.sh [full|dry-run]"
  exit 2
fi

python -c "import torch, peft; print('Shapley AdaLoRA dependencies OK')"
ARGS=()
[[ "$MODE" == "dry-run" ]] && ARGS+=(--dry-run)
python analysis/run_abr_c_matched_server2.py "${ARGS[@]}"
