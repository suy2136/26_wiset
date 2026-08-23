#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash scripts/run_nbs_adaptive_seed_budget_sequence.sh {smoke|full}"
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

if [[ "$MODE" == "smoke" ]]; then
  export EPOCHS=1
  export VALIDATION_INTERVAL=5
  export CHECKPOINT_INTERVAL=10
  export GRAD_ACCUM_STEPS=1
  export ADALORA_ALLOCATION_INTERVAL=1
  export LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-10}"
  export LIMIT_VALID_SAMPLES="${LIMIT_VALID_SAMPLES:-5}"
  export LIMIT_TEST_SAMPLES="${LIMIT_TEST_SAMPLES:-10}"
  echo "Smoke mode: 10 train / 5 valid / 10 test samples per experiment"
fi

echo "Verifying NBS allocator and independent seed controls"
python analysis/verify_nash_allocator.py
python analysis/verify_seed_separation.py

echo "[1/3] Adaptive NBS: min=2, max=32, floor=128, cap=512, tau=0.15, all seeds=1"
bash scripts/run_netllm_experiment.sh nbs_adaptive_tau015

echo "[2/3] Fixed NBS v19 data-seed ablation: budget=512, master/LoRA seed=1, data seed=2"
bash scripts/run_netllm_experiment.sh nbs_v19_data2

echo "[3/3] Fixed low-budget NBS: min=2, max=32, budget=256 (mean rank=4), all seeds=1"
bash scripts/run_netllm_experiment.sh nbs_budget256_seed1

echo "Adaptive, data-seed, and fixed low-budget NBS experiments completed."
