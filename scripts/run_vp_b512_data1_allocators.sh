#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash scripts/run_vp_b512_data1_allocators.sh {smoke|full}"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Capacity-matched VP comparison with NBS v19. Model, LoRA, and data seeds
# all remain at 1; each variant has an isolated experiment/model directory.
export EPOCHS=4
export VALIDATION_INTERVAL=4410
export CHECKPOINT_INTERVAL=8820
export SAVE_PERIODIC_CHECKPOINTS=0
export EVAL_PROGRESS_INTERVAL=500
export LATENCY_WARMUP_STEPS=5
export GRAD_ACCUM_STEPS=32
export LEARNING_RATE=0.0002
export SEED=1
export LORA_SEED=1
export DATA_SEED=1
export SKIP_VISUALIZATION="${SKIP_VISUALIZATION:-1}"

# EVA may not satisfy its strict convergence check within 128 batches. Keep
# the collected activation spectrum after a longer, bounded calibration run.
export EVA_MIN_BATCHES="${EVA_MIN_BATCHES:-2}"
export EVA_MAX_BATCHES="${EVA_MAX_BATCHES:-512}"
export EVA_SIMILARITY_THRESHOLD="${EVA_SIMILARITY_THRESHOLD:-0.99}"
export EVA_ALLOW_UNCONVERGED="${EVA_ALLOW_UNCONVERGED:-1}"

if [[ "$MODE" == "smoke" ]]; then
  export VP_B512_SMOKE=1
  export LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-10}"
  export LIMIT_VALID_SAMPLES="${LIMIT_VALID_SAMPLES:-5}"
  export LIMIT_TEST_SAMPLES="${LIMIT_TEST_SAMPLES:-10}"
  export EVA_LIMIT_TRAIN_SAMPLES="${EVA_LIMIT_TRAIN_SAMPLES:-10}"
  export EVA_MAX_BATCHES="${EVA_MAX_BATCHES_SMOKE:-3}"
  export EVA_SIMILARITY_THRESHOLD="${EVA_SIMILARITY_THRESHOLD_SMOKE:-0.0}"
  echo "VP budget-512/data-seed1 smoke: 10 train / 5 valid / 10 test samples"
fi

python -c "import torch, peft, torch_incremental_pca; print('VP allocator dependencies OK')"
python analysis/verify_seed_separation.py

echo "[1/3] Uniform LoRA: rank8, budget512, all seeds 1"
bash scripts/run_netllm_experiment.sh uniform_r8_data1

echo "[2/3] Stock AdaLoRA: init32, target8 (budget512), all seeds 1"
bash scripts/run_netllm_experiment.sh adalora_b512_data1

echo "[3/3] EVA: min2-max32, budget512, all seeds 1"
bash scripts/run_netllm_experiment.sh eva_b512_data1

echo "VP budget-512/data-seed1 allocator comparison completed."
