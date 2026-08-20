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

echo "Shared NBS conditions: min=4, max=32, budget=736, seed=1, direct AR evaluation"

echo "[1/5] Stock PEFT AdaLoRA r12: train once, then Direct AR and Selector + Speculative"
bash scripts/run_stock_adalora_full_stack.sh

echo "[2/5] B / NBS v15: lr=2e-4, EMA beta=0.95 (higher-beta sensitivity smoothing)"
bash scripts/run_netllm_experiment.sh nbs_v15

echo "[3/5] D / NBS v17: lr=2e-4, EMA beta=0.80 (lower-beta responsiveness)"
bash scripts/run_netllm_experiment.sh nbs_v17

echo "[4/5] A / NBS v14: lr=1.5e-4, EMA beta=0.90 (lower learning rate)"
bash scripts/run_netllm_experiment.sh nbs_v14

echo "[5/5] C / NBS v16: lr=2.5e-4, EMA beta=0.90 (higher learning rate)"
bash scripts/run_netllm_experiment.sh nbs_v16

echo "AdaLoRA and NBS hyperparameter sweep completed in B-D-A-C order."
