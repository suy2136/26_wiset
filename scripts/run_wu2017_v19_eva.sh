#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARTIFACT_ROOT="viewport_prediction/data/experiment_runs/netllm_vs_nbs"
NBS_RUN_DIR="${NBS_RUN_DIR:-$(tr -d '\r\n' < "$ARTIFACT_ROOT/nbs_v19_latest.txt")}"
EVA_RUN_DIR="${EVA_RUN_DIR:-$(tr -d '\r\n' < "$ARTIFACT_ROOT/eva_latest.txt")}"

metadata_value() {
  local file="$1"
  local key="$2"
  sed -n "s/^${key}=//p" "$file" | tail -n 1
}
NBS_CHECKPOINT="${NBS_CHECKPOINT:-$(metadata_value "$NBS_RUN_DIR/metadata.env" best_ar_model)}"
EVA_CHECKPOINT="${EVA_CHECKPOINT:-$(metadata_value "$EVA_RUN_DIR/metadata.env" best_ar_model)}"
if [[ -z "$EVA_CHECKPOINT" ]]; then
  EVA_CHECKPOINT="$(metadata_value "$EVA_RUN_DIR/metadata.env" best_model)"
fi
EVA_STATE="${EVA_STATE:-$(metadata_value "$EVA_RUN_DIR/metadata.env" eva_state)}"
for path in "$NBS_CHECKPOINT" "$EVA_CHECKPOINT" "$EVA_STATE"; do
  [[ -e "$path" ]] || { echo "Required Wu2017 evaluation input missing: $path"; exit 2; }
done

python analysis/verify_viewport_datasets.py --datasets Wu2017 --splits test --frequency 5
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="viewport_prediction/data/experiment_runs/unseen_wu2017_v19_eva/$RUN_ID"
mkdir -p "$RUN_DIR/nbs_v19" "$RUN_DIR/eva" "$RUN_DIR/comparison"
printf '%s\n' "$RUN_DIR" > viewport_prediction/data/experiment_runs/unseen_wu2017_v19_eva_latest.txt

LIMIT_ARGS=()
if [[ -n "${LIMIT_TEST_SAMPLES:-}" ]]; then
  LIMIT_ARGS+=(--limit-test-samples "$LIMIT_TEST_SAMPLES")
fi
COMMON_ARGS=(
  --test --train-dataset Jin2022 --test-dataset Wu2017
  --plm-type llama --plm-size base --device cuda --device-out cuda --fp16
  --epochs 4 --bs 1 --grad-accum-steps 32 --lr 0.0002 --seed 1
  --save-test-progress-per-steps "${EVAL_PROGRESS_INTERVAL:-500}"
  --measure-inference-latency --latency-warmup-steps "${LATENCY_WARMUP_STEPS:-5}"
  "${LIMIT_ARGS[@]}"
)

run_case() {
  local label="$1"
  local checkpoint="$2"
  local output="$3"
  shift 3
  echo "Evaluating $label on unseen Wu2017"
  python run_plm.py "${COMMON_ARGS[@]}" \
    --model-path "$checkpoint" \
    --results-output-dir "$output" \
    --latency-output-path "$output/latency.json" \
    "$@" 2>&1 | tee "$output/evaluation.log"
  local result
  result="$(find "$output" -maxdepth 1 -type f -name '*_results.csv' \
    ! -name '*_partial_results.csv' ! -name '*_per_sample_results.csv' | head -n 1)"
  [[ -n "$result" ]] || { echo "Result CSV missing for $label"; return 3; }
  cp "$result" "$output/results.csv"
  local detail="${result/_results.csv/_per_sample_results.csv}"
  [[ ! -f "$detail" ]] || cp "$detail" "$output/per_sample_results.csv"
}

run_case "NBS v19" "$NBS_CHECKPOINT" "$RUN_DIR/nbs_v19" \
  --rank 32 --use-adalora --adalora-allocator nbs \
  --adalora-rank-config configs/adalora_rank_config_llama7b_min2_max32.json \
  --adalora-rank-budget 512 --experiment-tag nbs_v19

run_case "EVA" "$EVA_CHECKPOINT" "$RUN_DIR/eva" \
  --rank 12 --use-eva --eva-state-path "$EVA_STATE" --experiment-tag eva

python analysis/compare_wu2017_nbs_eva.py \
  --nbs-dir "$RUN_DIR/nbs_v19" --eva-dir "$RUN_DIR/eva" \
  --output-dir "$RUN_DIR/comparison"
printf '{"status":"complete","run_id":"%s","run_dir":"%s"}\n' \
  "$RUN_ID" "$RUN_DIR" > "$RUN_DIR/status.json"
echo "NBS v19/EVA Wu2017 evaluation complete: $RUN_DIR"
