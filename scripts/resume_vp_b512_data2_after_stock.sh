#!/usr/bin/env bash
set -euo pipefail

# Recover the evaluation of an already-trained budget-512 Stock AdaLoRA run,
# then continue with EVA and Shapley. The Stock adapter is never retrained or
# overwritten. Pass `stock`, `eva`, or `shapley` to run only one stage.

MODE="${1:-all}"
if [[ "$MODE" != "all" && "$MODE" != "stock" && \
      "$MODE" != "eva" && "$MODE" != "shapley" ]]; then
  echo "Usage: bash scripts/resume_vp_b512_data2_after_stock.sh {all|stock|eva|shapley} [STOCK_RUN_DIR]"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LATEST_FILE="viewport_prediction/data/experiment_runs/netllm_vs_nbs/adalora_b512_data2_latest.txt"
STOCK_RUN_DIR="${2:-}"

metadata_value() {
  sed -n "s/^${1}=//p" "$STOCK_RUN_DIR/metadata.env" | tail -n 1
}

adapter_complete() {
  local path="$1"
  [[ -f "$path/adapter_config.json" && \
     -f "$path/modules_except_plm.bin" && \
     -f "$path/checkpoint_metadata.json" && \
     ( -f "$path/adapter_model.bin" || -f "$path/adapter_model.safetensors" ) ]]
}

recover_stock_evaluation() {
  if [[ -z "$STOCK_RUN_DIR" ]]; then
    if [[ ! -f "$LATEST_FILE" ]]; then
      echo "Stock run pointer was not found: $LATEST_FILE"
      exit 2
    fi
    STOCK_RUN_DIR="$(<"$LATEST_FILE")"
  fi
  STOCK_RUN_DIR="${STOCK_RUN_DIR%/}"
  if [[ ! -f "$STOCK_RUN_DIR/metadata.env" ]]; then
    echo "Stock run metadata was not found: $STOCK_RUN_DIR/metadata.env"
    exit 2
  fi

  local run_id seed lora_seed data_seed epochs rank lr grad_accum
  run_id="$(metadata_value run_id)"
  seed="$(metadata_value seed)"; seed="${seed:-1}"
  lora_seed="$(metadata_value lora_seed)"; lora_seed="${lora_seed:-1}"
  data_seed="$(metadata_value data_seed)"; data_seed="${data_seed:-2}"
  epochs="$(metadata_value epochs)"; epochs="${epochs:-4}"
  rank="$(metadata_value rank)"; rank="${rank:-8}"
  lr="$(metadata_value learning_rate)"; lr="${lr:-0.0002}"
  grad_accum="${GRAD_ACCUM_STEPS:-32}"
  if [[ -z "$run_id" ]]; then
    echo "run_id is missing from $STOCK_RUN_DIR/metadata.env"
    exit 2
  fi

  local candidates=()
  mapfile -t candidates < <(
    find viewport_prediction/data/ft_plms -type d \
      -path "*/${run_id}/*" -name best_ar_model 2>/dev/null | sort
  )
  if [[ "${#candidates[@]}" -ne 1 ]]; then
    echo "Expected exactly one Stock best_ar_model for run $run_id; found ${#candidates[@]}"
    printf '  %s\n' "${candidates[@]}"
    exit 3
  fi
  local model_path="${candidates[0]}"
  if ! adapter_complete "$model_path"; then
    echo "Stock AdaLoRA checkpoint is incomplete: $model_path"
    exit 3
  fi

  local role_dir="$STOCK_RUN_DIR/evaluations/best_ar"
  local generated_dir="$role_dir/generated_results"
  mkdir -p "$role_dir/figures" "$generated_dir"
  if [[ "${FORCE_REEVALUATE:-0}" == "1" || \
        ! -s "$role_dir/results.csv" || ! -s "$role_dir/latency.json" ]]; then
    echo "[Stock AdaLoRA] recovering best_ar evaluation from $model_path"
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python run_plm.py \
        --test --train-dataset Jin2022 --test-dataset Jin2022 \
        --plm-type llama --plm-size base --device cuda --device-out cuda --fp16 \
        --rank "$rank" --use-adalora --adalora-allocator peft \
        --adalora-init-rank 32 --adalora-allocation-interval 10 \
        --experiment-tag adalora_b512_data2 --experiment-run-id "$run_id" \
        --model-path "$model_path" --evaluation-tag best_ar \
        --results-output-dir "$generated_dir" \
        --epochs "$epochs" --bs 1 --grad-accum-steps "$grad_accum" \
        --lr "$lr" --seed "$seed" --lora-seed "$lora_seed" \
        --data-seed "$data_seed" --save-test-progress-per-steps 500 \
        --measure-inference-latency --latency-warmup-steps 5 \
        --latency-output-path "$role_dir/latency.json" \
        2>&1 | tee "$role_dir/test.log"

    local results=()
    mapfile -t results < <(
      find "$generated_dir" -maxdepth 1 -type f -name '*_results.csv' \
        ! -name '*_per_sample_results.csv' ! -name '*_partial_results.csv' | sort
    )
    if [[ "${#results[@]}" -ne 1 ]]; then
      echo "Expected one Stock aggregate result CSV; found ${#results[@]}"
      exit 4
    fi
    cp "${results[0]}" "$role_dir/results.csv"
    local per_sample="${results[0]/_results.csv/_per_sample_results.csv}"
    local predictions="${results[0]/_results.csv/_predictions.txt}"
    [[ -f "$per_sample" ]] && cp "$per_sample" "$role_dir/per_sample_results.csv"
    [[ -f "$predictions" ]] && cp "$predictions" "$role_dir/predictions.txt"
  else
    echo "[Stock AdaLoRA] existing evaluation found; inference skipped"
  fi

  python analysis/plot_netllm_experiment.py \
    --variant adalora_b512_data2 --train-log "$STOCK_RUN_DIR/train.log" \
    --result-csv "$role_dir/results.csv" --output-dir "$role_dir/figures" \
    --checkpoint-role best_ar --latency-json "$role_dir/latency.json" \
    --adapter-config "$model_path/adapter_config.json" \
    --display-name "Stock AdaLoRA (budget512, data seed2)" \
    2>&1 | tee "$role_dir/plot.log"

  cp "$role_dir/results.csv" "$STOCK_RUN_DIR/results.csv"
  cp "$role_dir/latency.json" "$STOCK_RUN_DIR/latency.json"
  [[ -f "$role_dir/per_sample_results.csv" ]] && \
    cp "$role_dir/per_sample_results.csv" "$STOCK_RUN_DIR/per_sample_results.csv"
  cp -a "$role_dir/figures/." "$STOCK_RUN_DIR/figures/"
  printf '{\n  "variant": "adalora_b512_data2",\n  "run_id": "%s",\n  "stage": "recovered_evaluation",\n  "status": "complete"\n}\n' \
    "$run_id" > "$STOCK_RUN_DIR/status.json"
  echo "[Stock AdaLoRA] evaluation recovery complete: $STOCK_RUN_DIR"
}

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

if [[ "$MODE" == "all" || "$MODE" == "stock" ]]; then
  recover_stock_evaluation
fi
if [[ "$MODE" == "all" || "$MODE" == "eva" ]]; then
  echo "[EVA] training and evaluation"
  bash scripts/run_netllm_experiment.sh eva_b512_data2
fi
if [[ "$MODE" == "all" || "$MODE" == "shapley" ]]; then
  echo "[Shapley] training and best/final evaluation"
  bash scripts/run_netllm_experiment.sh shapley_b512_data2
fi

echo "Requested VP budget-512 recovery stages completed."
