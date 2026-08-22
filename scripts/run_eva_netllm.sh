#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: bash scripts/run_eva_netllm.sh {smoke|full}"
  exit 2
fi

export EPOCHS="${EPOCHS:-4}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
export LEARNING_RATE="${LEARNING_RATE:-0.0002}"
export SEED="${SEED:-1}"
export EVA_RANK="${EVA_RANK:-12}"
export EVA_RANK_BUDGET="${EVA_RANK_BUDGET:-736}"
export EVA_MIN_RANK="${EVA_MIN_RANK:-0}"
export EVA_MAX_RANK="${EVA_MAX_RANK:-24}"
export EVA_RHO="${EVA_RHO:-2.0}"
export EVA_METRIC="${EVA_METRIC:-ratio}"

if [[ "$MODE" == "smoke" ]]; then
  export EPOCHS=1
  export VALIDATION_INTERVAL=5
  export CHECKPOINT_INTERVAL=10
  export GRAD_ACCUM_STEPS=1
  export LIMIT_TRAIN_SAMPLES="${LIMIT_TRAIN_SAMPLES:-10}"
  export LIMIT_VALID_SAMPLES="${LIMIT_VALID_SAMPLES:-5}"
  export LIMIT_TEST_SAMPLES="${LIMIT_TEST_SAMPLES:-10}"
  export EVA_LIMIT_TRAIN_SAMPLES="${EVA_LIMIT_TRAIN_SAMPLES:-10}"
  export EVA_MIN_BATCHES="${EVA_MIN_BATCHES:-2}"
  export EVA_MAX_BATCHES="${EVA_MAX_BATCHES:-3}"
  export EVA_SIMILARITY_THRESHOLD="${EVA_SIMILARITY_THRESHOLD:-0.0}"
  export EVA_ALLOW_UNCONVERGED="${EVA_ALLOW_UNCONVERGED:-1}"
  echo "EVA smoke: 10 train / 5 valid / 10 test samples"
else
  echo "EVA full experiment: activation PCA -> fixed-rank training -> evaluation"
fi

python -c "import torch, peft, torch_incremental_pca; print('EVA dependencies OK')"
bash scripts/run_netllm_experiment.sh eva
