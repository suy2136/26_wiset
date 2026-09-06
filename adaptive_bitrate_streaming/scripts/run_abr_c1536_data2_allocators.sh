#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
if [[ "$MODE" != "full" && "$MODE" != "dry-run" && "$MODE" != "resume" ]]; then
  echo "Usage: bash scripts/run_abr_c1536_data2_allocators.sh {full|dry-run|resume}"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ABR_ROOT"

if [[ "$MODE" != "dry-run" ]]; then
  python -c "import torch, peft, torch_incremental_pca; print('ABR allocator dependencies OK')"
fi

ARGS=(
  --nbs-rollback-min-lr 1e-5
  --nbs-skip-batch-at-rollback-lr-floor
  --continue-on-error
)
if [[ "$MODE" == "dry-run" ]]; then
  ARGS+=(--dry-run)
elif [[ "$MODE" == "resume" ]]; then
  ARGS+=(--resume)
fi

python analysis/run_abr_c1536_data2_allocators.py "${ARGS[@]}"
