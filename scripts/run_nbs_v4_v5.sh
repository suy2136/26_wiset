#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Match the completed v1-v3 runs while validating twice per epoch.
export EPOCHS="${EPOCHS:-4}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"

echo "Shared conditions: epochs=$EPOCHS, validation interval=$VALIDATION_INTERVAL, checkpoint interval=$CHECKPOINT_INTERVAL"

echo "[1/2] Starting NBS-NetLLM v4 (early stopping, teacher forcing)"
bash scripts/run_netllm_experiment.sh nbs_v4

echo "[2/2] Starting NBS-NetLLM v5 (early stopping, scheduled sampling=0.1)"
bash scripts/run_netllm_experiment.sh nbs_v5

echo "Both NBS v4/v5 experiments completed."
