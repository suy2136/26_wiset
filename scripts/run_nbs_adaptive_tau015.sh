#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash scripts/run_nbs_adaptive_tau015.sh {smoke|full}"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export EPOCHS="${EPOCHS:-4}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
export LATENCY_WARMUP_STEPS="${LATENCY_WARMUP_STEPS:-5}"
export LEARNING_RATE="${LEARNING_RATE:-0.0002}"
export ADALORA_EMA_BETA="${ADALORA_EMA_BETA:-0.9}"
export SEED="${SEED:-1}"

if [[ "$MODE" == "smoke" ]]; then
  export EPOCHS=1
  export VALIDATION_INTERVAL=5
  export CHECKPOINT_INTERVAL=10
  export GRAD_ACCUM_STEPS=1
  export ADALORA_ALLOCATION_INTERVAL=1
  export LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-10}"
  export LIMIT_VALID_SAMPLES="${LIMIT_VALID_SAMPLES:-5}"
  export LIMIT_TEST_SAMPLES="${LIMIT_TEST_SAMPLES:-10}"
  echo "Adaptive NBS tau=0.15 smoke: 10 train / 5 valid / 10 test samples"
else
  echo "Adaptive NBS tau=0.15 full: v19 conditions, min2-max32, floor128, cap512"
fi

python analysis/verify_nash_allocator.py
bash scripts/run_netllm_experiment.sh nbs_adaptive_tau015
