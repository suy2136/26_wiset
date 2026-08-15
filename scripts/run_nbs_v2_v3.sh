#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Match the completed v1/plain experiments unless explicitly overridden.
export EPOCHS="${EPOCHS:-4}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"

echo "Shared conditions: epochs=$EPOCHS, validation/checkpoint interval=$CHECKPOINT_INTERVAL"

echo "[1/2] Starting NBS-NetLLM v2 (min=16, max=48, budget=2048)"
bash scripts/run_netllm_experiment.sh nbs_v2

echo "[2/2] Starting NBS-NetLLM v3 (min=8, max=64, budget=1536)"
bash scripts/run_netllm_experiment.sh nbs_v3

echo "Both NBS experiments completed."
