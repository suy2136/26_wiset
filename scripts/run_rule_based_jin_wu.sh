#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python -c "import torch, sklearn; print('rule-based dependencies OK')"
python analysis/verify_viewport_datasets.py \
  --datasets Jin2022 Wu2017 --splits test --frequency 5

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="viewport_prediction/data/experiment_runs/rule_based_jin_wu/$RUN_ID"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > viewport_prediction/data/experiment_runs/rule_based_jin_wu_latest.txt
ACTIVE_CASE=""
ACTIVE_OUTPUT=""

write_status() {
  local path="$1"
  local payload="$2"
  printf '%s\n' "$payload" > "$path.tmp"
  mv "$path.tmp" "$path"
}

on_error() {
  local code=$?
  set +e
  if [[ -n "$ACTIVE_OUTPUT" ]]; then
    write_status "$ACTIVE_OUTPUT/status.json" \
      "{\"case\":\"$ACTIVE_CASE\",\"status\":\"failed\",\"exit_code\":$code}"
  fi
  write_status "$RUN_DIR/status.json" \
    "{\"run_id\":\"$RUN_ID\",\"status\":\"failed\",\"failed_case\":\"$ACTIVE_CASE\",\"exit_code\":$code,\"run_dir\":\"$RUN_DIR\"}"
  echo "Rule-based evaluation failed in $ACTIVE_CASE; completed artifacts remain in $RUN_DIR"
  exit "$code"
}
trap on_error ERR

LIMIT_ARGS=()
if [[ -n "${LIMIT_TEST_SAMPLES:-}" ]]; then
  LIMIT_ARGS=(--limit-test-samples "$LIMIT_TEST_SAMPLES")
fi
write_status "$RUN_DIR/status.json" \
  "{\"run_id\":\"$RUN_ID\",\"status\":\"running\",\"run_dir\":\"$RUN_DIR\"}"

run_case() {
  local number="$1"
  local model="$2"
  local dataset="$3"
  local output_dir="$RUN_DIR/${number}_${model}_${dataset}"
  mkdir -p "$output_dir"
  ACTIVE_CASE="${model}_${dataset}"
  ACTIVE_OUTPUT="$output_dir"
  write_status "$output_dir/status.json" \
    "{\"model\":\"$model\",\"dataset\":\"$dataset\",\"status\":\"running\"}"
  echo "[$number/4] $model on $dataset"

  python run_baseline.py \
    --test \
    --model "$model" \
    --train-dataset Jin2022 \
    --test-dataset "$dataset" \
    --device cpu \
    --bs 1 \
    --his-window 10 \
    --fut-window 20 \
    --dataset-frequency 5 \
    --sample-step 15 \
    --seed 1 \
    --measure-inference-latency \
    --latency-warmup-steps "${LATENCY_WARMUP_STEPS:-5}" \
    --latency-output-path "$output_dir/latency.json" \
    --results-output-dir "$output_dir" \
    "${LIMIT_ARGS[@]}" \
    2>&1 | tee "$output_dir/evaluation.log"

  local result_csv
  result_csv="$(find "$output_dir" -maxdepth 1 -type f -name 'result_*.csv' | head -1)"
  if [[ -z "$result_csv" ]]; then
    echo "Result CSV missing for $model on $dataset"
    exit 3
  fi
  cp "$result_csv" "$output_dir/results.csv"
  write_status "$output_dir/status.json" \
    "{\"model\":\"$model\",\"dataset\":\"$dataset\",\"status\":\"complete\"}"
  ACTIVE_CASE=""
  ACTIVE_OUTPUT=""
}

run_case 1 regression Jin2022
run_case 2 velocity Jin2022
run_case 3 regression Wu2017
run_case 4 velocity Wu2017

write_status "$RUN_DIR/status.json" \
  "{\"run_id\":\"$RUN_ID\",\"status\":\"complete\",\"run_dir\":\"$RUN_DIR\"}"
echo "Rule-based Jin2022/Wu2017 evaluation complete: $RUN_DIR"
