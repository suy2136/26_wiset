#!/usr/bin/env bash
set -u
set -o pipefail

# Evaluate and visualize an already trained Shapley run.  This script never
# resumes training and never rewrites either saved adapter checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DEFAULT_LATEST_FILE="viewport_prediction/data/experiment_runs/netllm_vs_nbs/shapley_v19_latest.txt"
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  if [[ ! -f "$DEFAULT_LATEST_FILE" ]]; then
    echo "Usage: bash scripts/resume_shapley_evaluation.sh RUN_DIR"
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
  sed -n "s/^${1}=//p" "$METADATA" | tail -n 1
}

RUN_ID="$(metadata_value run_id)"
SEED="$(metadata_value seed)"
LORA_SEED="$(metadata_value lora_seed)"
DATA_SEED="$(metadata_value data_seed)"
EPOCHS="$(metadata_value epochs)"
RANK="$(metadata_value rank)"
LEARNING_RATE="$(metadata_value learning_rate)"
EVAL_PROGRESS_INTERVAL="$(metadata_value eval_progress_interval)"
LATENCY_WARMUP_STEPS="$(metadata_value latency_warmup_steps)"
SEED="${SEED:-1}"
LORA_SEED="${LORA_SEED:-$SEED}"
DATA_SEED="${DATA_SEED:-$SEED}"
EPOCHS="${EPOCHS:-4}"
RANK="${RANK:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS_OVERRIDE:-32}"
EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
LATENCY_WARMUP_STEPS="${LATENCY_WARMUP_STEPS:-5}"

if [[ -z "$RUN_ID" ]]; then
  echo "run_id is missing from $METADATA"
  exit 2
fi

mapfile -t BEST_CANDIDATES < <(
  find viewport_prediction/data/ft_plms -type d \
    -path "*/${RUN_ID}/*" -name best_ar_model 2>/dev/null | sort
)
if [[ "${#BEST_CANDIDATES[@]}" -ne 1 ]]; then
  echo "Expected exactly one best_ar_model for run $RUN_ID; found ${#BEST_CANDIDATES[@]}"
  printf '  %s\n' "${BEST_CANDIDATES[@]}"
  exit 3
fi
BEST_MODEL="${BEST_CANDIDATES[0]}"
CHECKPOINT_ROOT="$(dirname "$BEST_MODEL")"
FINAL_MODEL="$CHECKPOINT_ROOT/final_shapley_model"

for model_path in "$BEST_MODEL" "$FINAL_MODEL"; do
  for file_name in adapter_model.bin adapter_config.json modules_except_plm.bin checkpoint_metadata.json; do
    if [[ ! -f "$model_path/$file_name" ]]; then
      echo "Incomplete Shapley checkpoint: $model_path/$file_name"
      exit 3
    fi
  done
done

write_status() {
  printf '{\n  "variant": "shapley_v19",\n  "display_name": "Recovered Shapley AdaLoRA",\n  "run_id": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "exit_code": %s,\n  "updated_at": "%s"\n}\n' \
    "$RUN_ID" "$1" "$2" "$3" "$(date --iso-8601=seconds)" > "$RUN_DIR/status.json"
}

run_logged() {
  local log_path="$1"
  shift
  set +e
  "$@" 2>&1 | tee -a "$log_path"
  local code=${PIPESTATUS[0]}
  set -e
  return "$code"
}

if [[ -f "$RUN_DIR/status.json" && ! -f "$RUN_DIR/status.before_shapley_recovery.json" ]]; then
  cp "$RUN_DIR/status.json" "$RUN_DIR/status.before_shapley_recovery.json"
fi

set -e
ROLES=(best_ar final_shapley)
MODELS=("$BEST_MODEL" "$FINAL_MODEL")
for index in "${!ROLES[@]}"; do
  ROLE="${ROLES[$index]}"
  MODEL_PATH="${MODELS[$index]}"
  ROLE_DIR="$RUN_DIR/evaluations/$ROLE"
  GENERATED_DIR="$ROLE_DIR/generated_results"
  mkdir -p "$ROLE_DIR/figures" "$GENERATED_DIR"

  if [[ "${FORCE_REEVALUATE:-0}" == "1" || ! -s "$ROLE_DIR/results.csv" || ! -s "$ROLE_DIR/latency.json" ]]; then
    write_status "recovery_evaluation_${ROLE}" "running" 0
    if run_logged "$ROLE_DIR/test.log" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python run_plm.py \
        --test --train-dataset Jin2022 --test-dataset Jin2022 \
        --plm-type llama --plm-size base --device cuda --device-out cuda --fp16 \
        --rank "$RANK" --use-adalora --adalora-allocator shapley \
        --experiment-tag adalora_shapley --experiment-run-id "$RUN_ID" \
        --model-path "$MODEL_PATH" --evaluation-tag "$ROLE" \
        --results-output-dir "$GENERATED_DIR" \
        --epochs "$EPOCHS" --bs 1 --grad-accum-steps "$GRAD_ACCUM_STEPS" \
        --lr "$LEARNING_RATE" --seed "$SEED" --lora-seed "$LORA_SEED" \
        --data-seed "$DATA_SEED" --save-test-progress-per-steps "$EVAL_PROGRESS_INTERVAL" \
        --measure-inference-latency --latency-warmup-steps "$LATENCY_WARMUP_STEPS" \
        --latency-output-path "$ROLE_DIR/latency.json"; then
      :
    else
      code=$?
      write_status "recovery_evaluation_${ROLE}" "failed" "$code"
      exit "$code"
    fi

    mapfile -t RESULT_CANDIDATES < <(
      find "$GENERATED_DIR" -maxdepth 1 -type f -name '*_results.csv' \
        ! -name '*_per_sample_results.csv' ! -name '*_partial_results.csv' | sort
    )
    if [[ "${#RESULT_CANDIDATES[@]}" -ne 1 ]]; then
      write_status "recovery_evaluation_${ROLE}" "failed_missing_results" 4
      echo "Expected one aggregate result CSV for $ROLE; found ${#RESULT_CANDIDATES[@]}"
      exit 4
    fi
    cp "${RESULT_CANDIDATES[0]}" "$ROLE_DIR/results.csv"
    PER_SAMPLE="${RESULT_CANDIDATES[0]/_results.csv/_per_sample_results.csv}"
    PREDICTIONS="${RESULT_CANDIDATES[0]/_results.csv/_predictions.txt}"
    [[ -f "$PER_SAMPLE" ]] && cp "$PER_SAMPLE" "$ROLE_DIR/per_sample_results.csv"
    [[ -f "$PREDICTIONS" ]] && cp "$PREDICTIONS" "$ROLE_DIR/predictions.txt"
  else
    echo "[$ROLE] existing results and latency found; inference skipped"
  fi

  write_status "recovery_visualization_${ROLE}" "running" 0
  run_logged "$ROLE_DIR/plot.log" python analysis/plot_netllm_experiment.py \
    --variant shapley_v19 --train-log "$RUN_DIR/train.log" \
    --result-csv "$ROLE_DIR/results.csv" --output-dir "$ROLE_DIR/figures" \
    --checkpoint-role "$ROLE" --latency-json "$ROLE_DIR/latency.json" \
    --adapter-config "$MODEL_PATH/adapter_config.json" \
    --display-name "Shapley AdaLoRA (early-stopped v19 conditions)"
done

write_status "recovery_complete" "complete" 0
echo "Shapley best/final evaluation and visualization complete: $RUN_DIR"
