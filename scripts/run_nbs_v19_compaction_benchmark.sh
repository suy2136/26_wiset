#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARTIFACT_ROOT="viewport_prediction/data/experiment_runs/netllm_vs_nbs"
NBS_VARIANT="${NBS_VARIANT:-nbs_v19}"
case "$NBS_VARIANT" in
  nbs_v19|nbs_v24|nbs_v25) ;;
  *) echo "NBS_VARIANT must be nbs_v19, nbs_v24, or nbs_v25"; exit 2 ;;
esac
V19_RUN_DIR="${V19_RUN_DIR:-}"
if [[ -z "$V19_RUN_DIR" ]]; then
  LATEST_POINTER="$ARTIFACT_ROOT/${NBS_VARIANT}_latest.txt"
  if [[ ! -f "$LATEST_POINTER" ]]; then
    echo "Set V19_RUN_DIR to the completed $NBS_VARIANT artifact directory."
    exit 2
  fi
  V19_RUN_DIR="$(<"$LATEST_POINTER")"
fi
if [[ ! -f "$V19_RUN_DIR/metadata.env" ]]; then
  echo "$NBS_VARIANT metadata.env not found: $V19_RUN_DIR"
  exit 2
fi

# The generated metadata contains simple path/numeric assignments only.
# shellcheck disable=SC1090
source "$V19_RUN_DIR/metadata.env"
CHECKPOINT_ROLE="${CHECKPOINT_ROLE:-best_ar}"
case "$CHECKPOINT_ROLE" in
  best_ar) SOURCE_CHECKPOINT="$best_ar_model" ;;
  best_post_nbs) SOURCE_CHECKPOINT="$best_post_nbs_model" ;;
  final_nbs) SOURCE_CHECKPOINT="$final_nbs_model" ;;
  *) echo "CHECKPOINT_ROLE must be best_ar, best_post_nbs, or final_nbs"; exit 2 ;;
esac
if [[ ! -d "$SOURCE_CHECKPOINT" ]]; then
  echo "Source NBS checkpoint not found: $SOURCE_CHECKPOINT"
  exit 3
fi

BENCHMARK_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ARTIFACT_ROOT/${NBS_VARIANT}_compaction/$BENCHMARK_ID"
ORIGINAL_DIR="$RUN_DIR/original"
COMPACT_DIR="$RUN_DIR/compact"
COMPACT_CHECKPOINT="$RUN_DIR/compact_checkpoint"
COMPARISON_DIR="$RUN_DIR/comparison"
mkdir -p "$ORIGINAL_DIR" "$COMPACT_DIR" "$COMPARISON_DIR"
printf '%s\n' "$RUN_DIR" > "$ARTIFACT_ROOT/${NBS_VARIANT}_compaction_latest.txt"

LIMIT_ARGS=()
if [[ -n "${LIMIT_TEST_SAMPLES:-}" ]]; then
  LIMIT_ARGS+=(--limit-test-samples "$LIMIT_TEST_SAMPLES")
fi
LATENCY_WARMUP_STEPS="${LATENCY_WARMUP_STEPS:-5}"
EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"

COMMON_ARGS=(
  --test
  --train-dataset Jin2022
  --test-dataset Jin2022
  --plm-type llama
  --plm-size base
  --device cuda
  --device-out cuda
  --fp16
  --rank 32
  --use-adalora
  --adalora-allocator nbs
  --adalora-rank-config configs/adalora_rank_config_llama7b_min2_max32.json
  --adalora-rank-budget "$rank_budget"
  --adalora-ema-beta "$adalora_ema_beta"
  --adalora-allocation-interval 10
  --experiment-tag "$NBS_VARIANT"
  --model-path "$SOURCE_CHECKPOINT"
  --evaluation-tag "$CHECKPOINT_ROLE"
  --epochs 4
  --bs 1
  --grad-accum-steps 32
  --lr "$learning_rate"
  --seed "$seed"
  --save-test-progress-per-steps "$EVAL_PROGRESS_INTERVAL"
  --measure-inference-latency
  --latency-warmup-steps "$LATENCY_WARMUP_STEPS"
  "${LIMIT_ARGS[@]}"
)

printf 'benchmark_id=%s\nsource_run=%s\nsource_checkpoint=%s\ncheckpoint_role=%s\n' \
  "$BENCHMARK_ID" "$V19_RUN_DIR" "$SOURCE_CHECKPOINT" "$CHECKPOINT_ROLE" \
  > "$RUN_DIR/metadata.env"

echo "[1/3] Original masked NBS AdaLoRA evaluation"
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_plm.py \
  "${COMMON_ARGS[@]}" \
  --nbs-inference-mode original \
  --results-output-dir "$ORIGINAL_DIR" \
  --latency-output-path "$ORIGINAL_DIR/latency.json" \
  2>&1 | tee "$ORIGINAL_DIR/test.log"

echo "[2/3] Compact fixed-LoRA conversion, equivalence check, and evaluation"
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python run_plm.py \
  "${COMMON_ARGS[@]}" \
  --nbs-inference-mode compact \
  --nbs-compact-output-dir "$COMPACT_CHECKPOINT" \
  --results-output-dir "$COMPACT_DIR" \
  --latency-output-path "$COMPACT_DIR/latency.json" \
  2>&1 | tee "$COMPACT_DIR/test.log"

normalize_results() {
  local directory="$1"
  local result
  local detail
  result="$(find "$directory" -maxdepth 1 -type f -name '*_results.csv' \
    ! -name '*_partial_results.csv' ! -name '*_per_sample_results.csv' | head -n 1)"
  detail="$(find "$directory" -maxdepth 1 -type f -name '*_per_sample_results.csv' | head -n 1)"
  [[ -n "$result" ]] || { echo "Evaluation results missing in $directory"; return 4; }
  cp "$result" "$directory/results.csv"
  [[ -z "$detail" ]] || cp "$detail" "$directory/per_sample_results.csv"
}
normalize_results "$ORIGINAL_DIR"
normalize_results "$COMPACT_DIR"

echo "[3/3] Accuracy, latency, topology, and equivalence comparison"
python analysis/compare_nbs_compaction.py \
  --original-dir "$ORIGINAL_DIR" \
  --compact-dir "$COMPACT_DIR" \
  --compact-checkpoint "$COMPACT_CHECKPOINT" \
  --output-dir "$COMPARISON_DIR" \
  --label "$NBS_VARIANT ($CHECKPOINT_ROLE)"

printf '{\n  "status": "complete",\n  "benchmark_id": "%s",\n  "source_checkpoint": "%s",\n  "compact_checkpoint": "%s"\n}\n' \
  "$BENCHMARK_ID" "$SOURCE_CHECKPOINT" "$COMPACT_CHECKPOINT" \
  > "$RUN_DIR/status.json"
echo "$NBS_VARIANT original/compact benchmark complete: $RUN_DIR"
