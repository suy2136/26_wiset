#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export EPOCHS="${EPOCHS:-4}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
export LATENCY_WARMUP_STEPS="${LATENCY_WARMUP_STEPS:-5}"
export SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
export LEARNING_RATE="${LEARNING_RATE:-0.0002}"
export ADALORA_EMA_BETA="${ADALORA_EMA_BETA:-0.9}"

echo "[2] Unseen Wu2017: compact NBS v19 vs EVA"
bash scripts/run_wu2017_v19_eva.sh

echo "[3a] NBS v22: min2/max32/budget384 (mean rank6), seed1"
bash scripts/run_netllm_experiment.sh nbs_v22

echo "[3b] NBS v23: min2/max32/budget320 (mean rank5), seed1"
bash scripts/run_netllm_experiment.sh nbs_v23

echo "[4a] NBS v24: v19 conditions, seed2"
bash scripts/run_netllm_experiment.sh nbs_v24

echo "[4b] NBS v25: v19 conditions, seed3"
bash scripts/run_netllm_experiment.sh nbs_v25

echo "[5a] Compact inference benchmark for v24/seed2 (no retraining)"
NBS_VARIANT=nbs_v24 CHECKPOINT_ROLE=best_ar \
  bash scripts/run_nbs_v19_compaction_benchmark.sh

echo "[5b] Compact inference benchmark for v25/seed3 (no retraining)"
NBS_VARIANT=nbs_v25 CHECKPOINT_ROLE=best_ar \
  bash scripts/run_nbs_v19_compaction_benchmark.sh

echo "Follow-up experiments 2-5 completed."
