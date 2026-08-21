#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export EPOCHS="${EPOCHS:-4}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
export LATENCY_WARMUP_STEPS="${LATENCY_WARMUP_STEPS:-5}"

echo "Shared conditions: lr=2e-4, EMA beta=0.9, epochs=$EPOCHS, direct AR evaluation"

echo "[1/3] NBS v18: min=2, max=32, budget=640 (mean rank=10), seed=1"
bash scripts/run_netllm_experiment.sh nbs_v18

echo "[2/3] NBS v19: min=2, max=32, budget=512 (mean rank=8), seed=1"
bash scripts/run_netllm_experiment.sh nbs_v19

echo "[3/3] NBS v20: v12 conditions min=4, max=32, budget=736, seed=2"
bash scripts/run_netllm_experiment.sh nbs_v20

echo "NBS v18/v19 budget ablations and v20 seed replication completed."
