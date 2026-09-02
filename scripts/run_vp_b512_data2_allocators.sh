#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash scripts/run_vp_b512_data2_allocators.sh {smoke|full}"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Capacity-matched VP comparison against the existing NBS v19 data-seed
# ablation. Only the data seed changes; model/LoRA initialization stays at 1.
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
export DATA_SEED=2
export SHAPLEY_PERMUTATIONS="${SHAPLEY_PERMUTATIONS:-1}"
export SHAPLEY_VALIDATION_BATCHES="${SHAPLEY_VALIDATION_BATCHES:-1}"
export SHAPLEY_TRUNCATE_FRACTION="${SHAPLEY_TRUNCATE_FRACTION:-0.05}"
export SHAPLEY_VALUE_MODE="${SHAPLEY_VALUE_MODE:-teacher-forcing}"

if [[ "$MODE" == "smoke" ]]; then
  export VP_B512_SMOKE=1
  export LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-10}"
  export LIMIT_VALID_SAMPLES="${LIMIT_VALID_SAMPLES:-5}"
  export LIMIT_TEST_SAMPLES="${LIMIT_TEST_SAMPLES:-10}"
  export EVA_LIMIT_TRAIN_SAMPLES="${EVA_LIMIT_TRAIN_SAMPLES:-10}"
  export EVA_MIN_BATCHES="${EVA_MIN_BATCHES:-2}"
  export EVA_MAX_BATCHES="${EVA_MAX_BATCHES:-3}"
  export EVA_SIMILARITY_THRESHOLD="${EVA_SIMILARITY_THRESHOLD:-0.0}"
  export EVA_ALLOW_UNCONVERGED=1
  echo "VP budget-512 smoke: 10 train / 5 valid / 10 test samples"
fi

python -c "import torch, peft, torch_incremental_pca; print('VP allocator dependencies OK')"
python analysis/verify_seed_separation.py

echo "[1/4] Uniform LoRA: rank8, budget512, LoRA seed1, data seed2"
bash scripts/run_netllm_experiment.sh uniform_r8_data2

echo "[2/4] Stock AdaLoRA: init32, target8 (budget512), LoRA seed1, data seed2"
bash scripts/run_netllm_experiment.sh adalora_b512_data2

echo "[3/4] EVA: min2-max32, budget512, LoRA seed1, data seed2"
bash scripts/run_netllm_experiment.sh eva_b512_data2

echo "[4/4] Shapley AdaLoRA: init32, target8 (budget512), LoRA seed1, data seed2"
bash scripts/run_netllm_experiment.sh shapley_b512_data2

echo "VP budget-512 data-seed2 allocator comparison completed."
