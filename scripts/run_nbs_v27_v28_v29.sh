#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
if [[ "$MODE" != "full" && "$MODE" != "smoke" ]]; then
  echo "Usage: bash scripts/run_nbs_v27_v28_v29.sh {smoke|full}"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Explicit v19 schedule. The matching variant definitions also set these
# values so direct and sequential invocations cannot fall back to 40 epochs.
export EPOCHS=4
export VALIDATION_INTERVAL=4410
export CHECKPOINT_INTERVAL=8820
export EVAL_PROGRESS_INTERVAL=500
export SAVE_PERIODIC_CHECKPOINTS=0
export GRAD_ACCUM_STEPS=32
export LEARNING_RATE=0.0002
export ADALORA_EMA_BETA=0.9
export ADALORA_ALLOCATION_INTERVAL=10

if [[ "$MODE" == "smoke" ]]; then
  export LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-10}"
  export LIMIT_VALID_SAMPLES="${LIMIT_VALID_SAMPLES:-5}"
  export LIMIT_TEST_SAMPLES="${LIMIT_TEST_SAMPLES:-10}"
  echo "Smoke mode: train=10, valid=5, test=10 samples per experiment"
fi

echo "Shared v19 schedule: epochs=4, validation=4410, checkpoint=8820, effective batch=32, lr=2e-4, EMA=0.9, seeds=1"

echo "[1/3] NBS v27: adaptive tau=0.05, floor=128, cap=512, active-only shadow"
bash scripts/run_netllm_experiment.sh nbs_v27

echo "[2/3] NBS v28: fixed budget=512, active-only shadow (v19 legacy path preserved)"
bash scripts/run_netllm_experiment.sh nbs_v28

echo "[3/3] NBS v29: fixed budget=192, mean rank=3, legacy shadow"
bash scripts/run_netllm_experiment.sh nbs_v29

echo "NBS v27-v29 sequence completed."
