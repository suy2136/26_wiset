#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export EPOCHS="${EPOCHS:-4}"
export VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-4410}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-8820}"
export SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
export EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
export LATENCY_WARMUP_STEPS="${LATENCY_WARMUP_STEPS:-5}"

echo "Stock PEFT AdaLoRA target-rank12 training + full-stack evaluation"
bash scripts/run_netllm_experiment.sh adalora_peft_r12

RUN_POINTER="viewport_prediction/data/experiment_runs/netllm_vs_nbs/adalora_peft_r12_latest.txt"
if [[ ! -f "$RUN_POINTER" ]]; then
  echo "AdaLoRA run pointer missing: $RUN_POINTER"
  exit 2
fi
RUN_DIR="$(tr -d '\r\n' < "$RUN_POINTER")"
METADATA="$RUN_DIR/metadata.env"
if [[ ! -f "$METADATA" ]]; then
  echo "AdaLoRA metadata missing: $METADATA"
  exit 2
fi
MODEL_PATH="$(sed -n 's/^best_ar_model=//p' "$METADATA")"
RUN_ID="$(basename "$RUN_DIR")"
if [[ ! -f "$MODEL_PATH/adapter_model.bin" || ! -f "$MODEL_PATH/modules_except_plm.bin" ]]; then
  echo "AdaLoRA checkpoint is incomplete: $MODEL_PATH"
  exit 2
fi

DIRECT_DIR="$RUN_DIR/direct_ar"
mkdir -p "$DIRECT_DIR/figures"
echo "Stock PEFT AdaLoRA direct autoregressive evaluation started"
python run_plm.py \
  --test \
  --train-dataset Jin2022 \
  --test-dataset Jin2022 \
  --plm-type llama \
  --plm-size base \
  --device cuda \
  --device-out cuda \
  --fp16 \
  --rank 12 \
  --use-adalora \
  --adalora-allocator peft \
  --experiment-tag adalora_peft_r12 \
  --experiment-run-id "$RUN_ID" \
  --model-path "$MODEL_PATH" \
  --epochs "$EPOCHS" \
  --bs 1 \
  --grad-accum-steps 32 \
  --lr 0.0002 \
  --seed 1 \
  --save-test-progress-per-steps "$EVAL_PROGRESS_INTERVAL" \
  --measure-inference-latency \
  --latency-warmup-steps "$LATENCY_WARMUP_STEPS" \
  --latency-output-path "$DIRECT_DIR/latency.json" \
  --results-output-dir "$DIRECT_DIR" \
  2>&1 | tee "$DIRECT_DIR/evaluation.log"

DIRECT_RESULT="$(find "$DIRECT_DIR" -maxdepth 1 -type f -name '*_results.csv' \
  ! -name '*_partial_results.csv' ! -name '*_per_sample_results.csv' | head -1)"
if [[ -z "$DIRECT_RESULT" ]]; then
  echo "Direct autoregressive result CSV is missing."
  exit 3
fi
cp "$DIRECT_RESULT" "$DIRECT_DIR/results.csv"
DIRECT_PER_SAMPLE="${DIRECT_RESULT/_results.csv/_per_sample_results.csv}"
[[ -f "$DIRECT_PER_SAMPLE" ]] && cp "$DIRECT_PER_SAMPLE" "$DIRECT_DIR/per_sample_results.csv"

python analysis/plot_netllm_experiment.py \
  --variant adalora_peft_r12 \
  --display-name "Stock PEFT AdaLoRA r12 - Direct AR" \
  --train-log "$RUN_DIR/train.log" \
  --result-csv "$DIRECT_DIR/results.csv" \
  --output-dir "$DIRECT_DIR/figures" \
  --checkpoint-role best \
  --latency-json "$DIRECT_DIR/latency.json" \
  2>&1 | tee "$DIRECT_DIR/plot.log"

python analysis/compare_adalora_inference_modes.py \
  --direct-dir "$DIRECT_DIR" \
  --full-stack-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/comparison"

printf '{"run_id":"%s","status":"complete","direct_ar":"%s","full_stack":"%s"}\n' \
  "$RUN_ID" "$DIRECT_DIR" "$RUN_DIR" > "$RUN_DIR/dual_inference_status.json"
echo "AdaLoRA direct AR and Selector + Speculative evaluations complete: $RUN_DIR"
