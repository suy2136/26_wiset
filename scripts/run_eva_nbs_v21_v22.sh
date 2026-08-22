#!/usr/bin/env bash
set -euo pipefail

export EPOCHS="${EPOCHS:-4}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
export SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
export LEARNING_RATE="${LEARNING_RATE:-0.0002}"
export ADALORA_EMA_BETA="${ADALORA_EMA_BETA:-0.9}"
export SEED="${SEED:-1}"

echo "[1/3] EVA-NetLLM (activation PCA, budget736, seed1)"
bash scripts/run_eva_netllm.sh full

echo "[2/3] NBS v21 (min2, max32, budget448, mean rank7, seed1)"
bash scripts/run_netllm_experiment.sh nbs_v21

echo "[3/3] NBS v22 (min2, max32, budget384, mean rank6, seed1)"
bash scripts/run_netllm_experiment.sh nbs_v22

echo "EVA, NBS v21, and NBS v22 experiments completed."
