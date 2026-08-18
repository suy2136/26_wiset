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

echo "Shared conditions: epochs=$EPOCHS, validation interval=$VALIDATION_INTERVAL, latency warm-up=$LATENCY_WARMUP_STEPS"

echo "[1/3] Uniform-rank NetLLM (rank=12, total active rank=768)"
bash scripts/run_netllm_experiment.sh uniform_r12

echo "[2/3] NBS-NetLLM v11 (min=2, max=32, budget=768)"
bash scripts/run_netllm_experiment.sh nbs_v11

echo "[3/3] NBS-NetLLM v12 (min=4, max=32, budget=736)"
bash scripts/run_netllm_experiment.sh nbs_v12

echo "Uniform-r12 and NBS v11/v12 experiments completed."
