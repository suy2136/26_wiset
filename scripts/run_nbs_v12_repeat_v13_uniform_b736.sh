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

echo "Shared conditions: seed=1, epochs=$EPOCHS, validation interval=$VALIDATION_INTERVAL, latency warm-up=$LATENCY_WARMUP_STEPS"

echo "[1/3] Repeating NBS v12 (min=4, max=32, budget=736, seed=1)"
bash scripts/run_netllm_experiment.sh nbs_v12_repeat

echo "[2/3] Starting NBS v13 (min=4, max=32, budget=720, seed=1)"
bash scripts/run_netllm_experiment.sh nbs_v13

echo "[3/3] Starting fixed near-uniform LoRA baseline (32x rank11 + 32x rank12, budget=736, seed=1)"
bash scripts/run_netllm_experiment.sh uniform_b736

echo "NBS v12 repeat, NBS v13, and fixed near-uniform budget-736 experiments completed."
