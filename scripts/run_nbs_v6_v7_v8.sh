#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Match v4 so rank bounds and total budget are the only changed conditions.
export EPOCHS="${EPOCHS:-4}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"

echo "Shared conditions: epochs=$EPOCHS, validation interval=$VALIDATION_INTERVAL, checkpoint interval=$CHECKPOINT_INTERVAL"

echo "[1/3] Starting NBS-NetLLM v6 (min=4, max=32, budget=1024)"
bash scripts/run_netllm_experiment.sh nbs_v6

echo "[2/3] Starting NBS-NetLLM v7 (min=8, max=32, budget=1024)"
bash scripts/run_netllm_experiment.sh nbs_v7

echo "[3/3] Starting NBS-NetLLM v8 (min=8, max=32, budget=1280)"
bash scripts/run_netllm_experiment.sh nbs_v8

echo "All NBS v6/v7/v8 capacity-ablation experiments completed."
