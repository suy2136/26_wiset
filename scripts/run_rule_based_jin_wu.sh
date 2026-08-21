#!/usr/bin/env bash
set -euo pipefail

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

run_case() {
  local number="$1"
  local model="$2"
  local dataset="$3"
  local output_dir="$RUN_DIR/${number}_${model}_${dataset}"
  mkdir -p "$output_dir"
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
    2>&1 | tee "$output_dir/evaluation.log"

  local result_csv
  result_csv="$(find "$output_dir" -maxdepth 1 -type f -name 'result_*.csv' | head -1)"
  if [[ -z "$result_csv" ]]; then
    echo "Result CSV missing for $model on $dataset"
    exit 3
  fi
  cp "$result_csv" "$output_dir/results.csv"
  printf '{"model":"%s","dataset":"%s","status":"complete"}\n' \
    "$model" "$dataset" > "$output_dir/status.json"
}

run_case 1 regression Jin2022
run_case 2 velocity Jin2022
run_case 3 regression Wu2017
run_case 4 velocity Wu2017

printf '{"run_id":"%s","status":"complete","run_dir":"%s"}\n' \
  "$RUN_ID" "$RUN_DIR" > "$RUN_DIR/status.json"
echo "Rule-based Jin2022/Wu2017 evaluation complete: $RUN_DIR"
