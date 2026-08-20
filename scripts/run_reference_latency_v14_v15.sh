#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

REFERENCE_REPO="${REFERENCE_REPO:-/workspace/26_wiset_reference}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-/workspace/26_wiset/viewport_prediction/data/ft_plms/llama_base_low_rank/freeze_plm_False/multimodal_none/Jin2022/5Hz/his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False/best_model}"
REFERENCE_MAX_SAMPLES="${REFERENCE_MAX_SAMPLES:-0}"

export EPOCHS="${EPOCHS:-4}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
export LATENCY_WARMUP_STEPS="${LATENCY_WARMUP_STEPS:-5}"

if [[ ! -d "$REFERENCE_REPO/.git" ]]; then
  echo "Reference repository not found at $REFERENCE_REPO"
  exit 2
fi
if [[ ! -d "$REFERENCE_MODEL_PATH" ]]; then
  echo "Reference plain-LoRA checkpoint not found at $REFERENCE_MODEL_PATH"
  exit 2
fi

REFERENCE_RUN_ID="$(date +%Y%m%d_%H%M%S)"
REFERENCE_RUN_DIR="viewport_prediction/data/experiment_runs/netllm_vs_nbs/reference_original_ar/$REFERENCE_RUN_ID"
mkdir -p "$REFERENCE_RUN_DIR"

echo "[1/3] Reference repository direct autoregressive latency"
echo "Reference repo: $REFERENCE_REPO"
echo "Checkpoint: $REFERENCE_MODEL_PATH"
python analysis/benchmark_reference_autoregressive.py \
  --reference-repo "$REFERENCE_REPO" \
  --model-path "$REFERENCE_MODEL_PATH" \
  --output "$REPO_ROOT/$REFERENCE_RUN_DIR/latency.json" \
  --warmup-steps "$LATENCY_WARMUP_STEPS" \
  --max-samples "$REFERENCE_MAX_SAMPLES" \
  2>&1 | tee "$REFERENCE_RUN_DIR/benchmark.log"

echo "[2/3] NBS v14: min=4, max=32, budget=736, lr=1.5e-4, EMA beta=0.9"
bash scripts/run_netllm_experiment.sh nbs_v14

echo "[3/3] NBS v15: min=4, max=32, budget=736, lr=2e-4, EMA beta=0.95"
bash scripts/run_netllm_experiment.sh nbs_v15

echo "Reference latency, NBS v14, and NBS v15 experiments completed."
