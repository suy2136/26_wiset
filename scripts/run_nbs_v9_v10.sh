#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Keep the v7/v8 conditions fixed so global rank budget is the only variable.
export EPOCHS="${EPOCHS:-4}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"

echo "Shared conditions: epochs=$EPOCHS, validation interval=$VALIDATION_INTERVAL, checkpoint interval=$CHECKPOINT_INTERVAL"

echo "[1/2] Starting NBS-NetLLM v9 (min=4, max=32, budget=896, initial mean rank=14)"
bash scripts/run_netllm_experiment.sh nbs_v9

echo "[2/2] Starting NBS-NetLLM v10 (min=4, max=32, budget=640, initial mean rank=10)"
bash scripts/run_netllm_experiment.sh nbs_v10

echo "Both NBS v9/v10 rank-budget experiments completed."
