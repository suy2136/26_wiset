#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
cd "$(dirname "$0")/.."

ARGS=(
  --fp16-numeric-safeguards
  --fp16-selective-clamp
  --fp16-selective-clamp-threshold 60000
  --skip-nonfinite-batches
  --nbs-skip-batch-at-rollback-lr-floor
  --nbs-rollback-min-lr 1e-5
  --continue-on-error
)

case "$MODE" in
  full) ;;
  resume) ARGS+=(--resume) ;;
  dry-run) ARGS+=(--dry-run) ;;
  *) echo "usage: $0 [full|resume|dry-run]" >&2; exit 2 ;;
esac

python analysis/run_abr_c1536_data2_safeguarded_allocators.py "${ARGS[@]}"
