#!/usr/bin/env bash
set -u
set -o pipefail

# Recover an NBS run that finished training but stopped before final_nbs
# evaluation (for example, because an earlier visualization command failed).
# This script never trains or rewrites the saved final_nbs_model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEFAULT_LATEST_FILE="viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_adaptive_tau015_latest.txt"
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  if [[ ! -f "$DEFAULT_LATEST_FILE" ]]; then
    echo "Usage: bash scripts/resume_nbs_final_evaluation.sh RUN_DIR"
    echo "Default latest-run pointer was not found: $DEFAULT_LATEST_FILE"
    exit 2
  fi
  RUN_DIR="$(<"$DEFAULT_LATEST_FILE")"
fi
RUN_DIR="${RUN_DIR%/}"

METADATA="$RUN_DIR/metadata.env"
if [[ ! -f "$METADATA" ]]; then
  echo "Run metadata was not found: $METADATA"
  exit 2
fi

metadata_value() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}=//p" "$METADATA" | tail -n 1)"
  printf '%s' "$value"
}

required_metadata() {
  local key="$1"
  local value
  value="$(metadata_value "$key")"
  if [[ -z "$value" ]]; then
    echo "Required metadata key is missing or empty: $key" >&2
    return 2
  fi
  printf '%s' "$value"
}

VARIANT="$(required_metadata variant)" || exit $?
RUN_ID="$(required_metadata run_id)" || exit $?
SEED="$(required_metadata seed)" || exit $?
LORA_SEED="$(required_metadata lora_seed)" || exit $?
DATA_SEED="$(required_metadata data_seed)" || exit $?
EPOCHS="$(required_metadata epochs)" || exit $?
RANK="$(required_metadata rank)" || exit $?
LEARNING_RATE="$(required_metadata learning_rate)" || exit $?
FINAL_MODEL="$(required_metadata final_nbs_model)" || exit $?
BEST_RESULT_CSV="$(required_metadata result_csv)" || exit $?
NBS_DIAGNOSTICS="$(required_metadata nbs_diagnostics)" || exit $?
RANK_CONFIG="$(required_metadata rank_config)" || exit $?
RANK_BUDGET="$(required_metadata rank_budget)" || exit $?

EMA_BETA="$(metadata_value adalora_ema_beta)"
EMA_BETA="${EMA_BETA:-0.9}"
SHADOW_POLICY="$(metadata_value adalora_shadow_update_policy)"
SHADOW_POLICY="${SHADOW_POLICY:-legacy}"
BUDGET_MODE="$(metadata_value adalora_budget_mode)"
BUDGET_MODE="${BUDGET_MODE:-fixed}"
RELATIVE_LAMBDA="$(metadata_value adalora_relative_lambda)"
RELATIVE_LAMBDA="${RELATIVE_LAMBDA:-0.15}"
MIN_BUDGET="$(metadata_value adalora_adaptive_min_budget)"
MAX_BUDGET="$(metadata_value adalora_adaptive_max_budget)"
EVAL_PROGRESS_INTERVAL="$(metadata_value eval_progress_interval)"
EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
LATENCY_WARMUP_STEPS="$(metadata_value latency_warmup_steps)"
LATENCY_WARMUP_STEPS="${LATENCY_WARMUP_STEPS:-5}"
if [[ "$FINAL_MODEL" =~ _bs_([0-9]+)_ ]]; then
  SAVED_GRAD_ACCUM_STEPS="${BASH_REMATCH[1]}"
else
  SAVED_GRAD_ACCUM_STEPS=32
fi
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$SAVED_GRAD_ACCUM_STEPS}"
DISPLAY_NAME="${DISPLAY_NAME:-Recovered ${VARIANT} final NBS}"

if [[ "$BEST_RESULT_CSV" != *"_checkpoint_best_ar"* ]]; then
  echo "Cannot derive final_nbs result path from: $BEST_RESULT_CSV"
  exit 2
fi
FINAL_RESULT_CSV="${BEST_RESULT_CSV/_checkpoint_best_ar/_checkpoint_final_nbs}"
FINAL_PARTIAL_CSV="${FINAL_RESULT_CSV/_results.csv/_partial_results.csv}"
FINAL_PREDICTIONS="${FINAL_RESULT_CSV/_results.csv/_predictions.txt}"
FINAL_PER_SAMPLE="${FINAL_RESULT_CSV/_results.csv/_per_sample_results.csv}"
FINAL_LATENCY="${FINAL_RESULT_CSV/_results.csv/_latency.json}"
FINAL_LATENCY_DETAIL="${FINAL_LATENCY%.json}_per_sample.csv"
ROLE_DIR="$RUN_DIR/evaluations/final_nbs"
mkdir -p "$RUN_DIR/figures" "$ROLE_DIR/figures" "$(dirname "$FINAL_RESULT_CSV")"

CANONICAL_MODEL="$(python analysis/resolve_checkpoint_alias.py "$FINAL_MODEL")" || exit $?
for required_file in adapter_model.bin modules_except_plm.bin nash_rank_allocator.pt; do
  if [[ ! -f "$CANONICAL_MODEL/$required_file" ]]; then
    echo "Final NBS checkpoint is incomplete: $CANONICAL_MODEL/$required_file"
    exit 3
  fi
done
if [[ ! -f "$RUN_DIR/train.log" || ! -f "$NBS_DIAGNOSTICS" ]]; then
  echo "Training log or allocator diagnostics are missing from $RUN_DIR"
  exit 3
fi

write_status() {
  local stage="$1"
  local state="$2"
  local code="$3"
  printf '{\n  "variant": "%s",\n  "display_name": "%s",\n  "run_id": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "exit_code": %s,\n  "updated_at": "%s"\n}\n' \
    "$VARIANT" "$DISPLAY_NAME" "$RUN_ID" "$stage" "$state" "$code" "$(date --iso-8601=seconds)" \
    > "$RUN_DIR/status.json"
}

run_logged() {
  local log_path="$1"
  shift
  set +e
  "$@" 2>&1 | tee -a "$log_path"
  local exit_code=${PIPESTATUS[0]}
  set -e
  return "$exit_code"
}

if [[ -f "$RUN_DIR/status.json" && ! -f "$RUN_DIR/status.before_final_recovery.json" ]]; then
  cp "$RUN_DIR/status.json" "$RUN_DIR/status.before_final_recovery.json"
fi

ADAPTIVE_ARGS=()
if [[ "$BUDGET_MODE" == "adaptive" ]]; then
  if [[ -z "$MIN_BUDGET" || -z "$MAX_BUDGET" ]]; then
    echo "Adaptive run metadata is missing its minimum or maximum budget."
    exit 2
  fi
  ADAPTIVE_ARGS=(
    --adalora-budget-mode adaptive
    --adalora-relative-lambda "$RELATIVE_LAMBDA"
    --adalora-adaptive-min-budget "$MIN_BUDGET"
    --adalora-adaptive-max-budget "$MAX_BUDGET"
  )
fi

set -e
if [[ "${FORCE_REEVALUATE:-0}" == "1" || ! -s "$FINAL_RESULT_CSV" || ! -s "$FINAL_LATENCY" ]]; then
  write_status "recovery_evaluation_final_nbs" "running" 0
  TEST_CMD=(
    python run_plm.py
    --test
    --train-dataset Jin2022
    --test-dataset Jin2022
    --plm-type llama
    --plm-size base
    --device cuda
    --device-out cuda
    --fp16
    --rank "$RANK"
    --experiment-run-id "$RUN_ID"
    --model-path "$FINAL_MODEL"
    --evaluation-tag final_nbs
    --epochs "$EPOCHS"
    --bs 1
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
    --lr "$LEARNING_RATE"
    --save-test-progress-per-steps "$EVAL_PROGRESS_INTERVAL"
    --measure-inference-latency
    --latency-warmup-steps "$LATENCY_WARMUP_STEPS"
    --latency-output-path "$FINAL_LATENCY"
    --seed "$SEED"
    --lora-seed "$LORA_SEED"
    --data-seed "$DATA_SEED"
    --use-adalora
    --adalora-allocator nbs
    --adalora-rank-config "$RANK_CONFIG"
    --adalora-rank-budget "$RANK_BUDGET"
    --adalora-ema-beta "$EMA_BETA"
    --adalora-shadow-update-policy "$SHADOW_POLICY"
    --adalora-diagnostics-path "$NBS_DIAGNOSTICS"
    --experiment-tag "$VARIANT"
    "${ADAPTIVE_ARGS[@]}"
  )
  if run_logged "$ROLE_DIR/test.log" \
      env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${TEST_CMD[@]}"; then
    :
  else
    code=$?
    write_status "recovery_evaluation_final_nbs" "failed" "$code"
    exit "$code"
  fi
else
  echo "Existing final_nbs results and latency are complete; inference skipped."
fi

if [[ ! -s "$FINAL_RESULT_CSV" || ! -s "$FINAL_LATENCY" ]]; then
  write_status "recovery_evaluation_final_nbs" "failed_missing_outputs" 4
  echo "Final evaluation did not produce both results and latency outputs."
  exit 4
fi

cp "$FINAL_RESULT_CSV" "$ROLE_DIR/results.csv"
cp "$FINAL_LATENCY" "$ROLE_DIR/latency.json"
[[ -f "$FINAL_PARTIAL_CSV" ]] && cp "$FINAL_PARTIAL_CSV" "$ROLE_DIR/partial_results.csv"
[[ -f "$FINAL_PREDICTIONS" ]] && cp "$FINAL_PREDICTIONS" "$ROLE_DIR/predictions.txt"
[[ -f "$FINAL_PER_SAMPLE" ]] && cp "$FINAL_PER_SAMPLE" "$ROLE_DIR/per_sample_results.csv"
[[ -f "$FINAL_LATENCY_DETAIL" ]] && cp "$FINAL_LATENCY_DETAIL" "$ROLE_DIR/latency_per_sample.csv"

write_status "recovery_visualization_final_nbs" "running" 0
PLOT_CMD=(
  python analysis/plot_netllm_experiment.py
  --variant "$VARIANT"
  --train-log "$RUN_DIR/train.log"
  --result-csv "$ROLE_DIR/results.csv"
  --output-dir "$ROLE_DIR/figures"
  --checkpoint-role final_nbs
  --latency-json "$ROLE_DIR/latency.json"
  --allocator-state "$CANONICAL_MODEL/nash_rank_allocator.pt"
  --allocator-diagnostics "$NBS_DIAGNOSTICS"
  --display-name "$DISPLAY_NAME"
)
if run_logged "$ROLE_DIR/plot.log" "${PLOT_CMD[@]}"; then
  :
else
  code=$?
  write_status "recovery_visualization_final_nbs" "failed" "$code"
  exit "$code"
fi

if [[ "$BUDGET_MODE" == "adaptive" ]]; then
  if run_logged "$RUN_DIR/adaptive_plot.log" \
      python analysis/plot_nbs_adaptive_budget.py \
        --diagnostics "$NBS_DIAGNOSTICS" \
        --output-dir "$RUN_DIR/figures" \
        --label "$DISPLAY_NAME"; then
    :
  else
    code=$?
    write_status "recovery_adaptive_visualization" "failed" "$code"
    exit "$code"
  fi
fi

if run_logged "$RUN_DIR/checkpoint_report.log" \
    python analysis/plot_nbs_checkpoint_report.py \
      --run-dir "$RUN_DIR" \
      --output-dir "$RUN_DIR/checkpoint_nbs_report" \
      --label "$DISPLAY_NAME"; then
  :
else
  code=$?
  write_status "recovery_checkpoint_report" "failed" "$code"
  exit "$code"
fi

write_status "recovery_complete" "complete" 0
echo "Final NBS evaluation and visualization complete: $ROLE_DIR"
