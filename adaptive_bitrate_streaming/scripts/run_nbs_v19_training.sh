#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/workspace/26_wiset/downloaded_plms/llama/base}"
EXP_POOL_PATH="${EXP_POOL_PATH:-$ABR_ROOT/artifacts/exp_pools/exp_pool.pkl}"
TRACE="${TRACE:-fcc-valid}"
TRACE_NUM="${TRACE_NUM:-100}"
VIDEO="${VIDEO:-video1}"
NUM_EPOCHS="${NUM_EPOCHS:-80}"
EVAL_PER_EPOCH="${EVAL_PER_EPOCH:-2}"

cd "$ABR_ROOT"

"$PYTHON_BIN" run_plm.py \
  --adapt --nbs-v19 --fp16 \
  --seed 1 \
  --plm-type llama --plm-size base --plm-dir "$BASE_MODEL_DIR" \
  --rank 32 --nbs-rank-budget 512 \
  --nbs-rank-config configs/nbs_v19_rank_config.json \
  --token-selector none --speculative-draft-steps 0 \
  --exp-pool-path "$EXP_POOL_PATH" \
  --trace "$TRACE" --trace-num "$TRACE_NUM" --video "$VIDEO" --fixed-order \
  --device cuda:0 --device-out cuda:0 \
  --grad-accum-steps 32 \
  --lr 0.0001 --warmup-steps 2000 \
  --num-epochs "$NUM_EPOCHS" --eval-per-epoch "$EVAL_PER_EPOCH"
