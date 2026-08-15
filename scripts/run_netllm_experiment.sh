#!/usr/bin/env bash
set -u
set -o pipefail

VARIANT="${1:-}"
if [[ "$VARIANT" != "nbs" && "$VARIANT" != "plain" ]]; then
  echo "Usage: bash scripts/run_netllm_experiment.sh {nbs|plain}"
  exit 2
fi

EPOCHS="${EPOCHS:-40}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$CHECKPOINT_INTERVAL" =~ ^[1-9][0-9]*$ && \
        "$EVAL_PROGRESS_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS, CHECKPOINT_INTERVAL, and EVAL_PROGRESS_INTERVAL must be positive integers."
  exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_ROOT="viewport_prediction/data/experiment_runs/netllm_vs_nbs"
RUN_DIR="$ARTIFACT_ROOT/$VARIANT/$RUN_ID"
mkdir -p "$RUN_DIR/figures"
printf '%s\n' "$RUN_DIR" > "$ARTIFACT_ROOT/${VARIANT}_latest.txt"

if [[ "$VARIANT" == "nbs" ]]; then
  MODEL_TAG="llama_base_low_rank_adalora"
  DISPLAY_NAME="NBS-NetLLM"
  NBS_DIAGNOSTICS="$RUN_DIR/nbs_rank_diagnostics.csv"
  EXTRA_ARGS=(
    --use-adalora
    --adalora-rank-config configs/adalora_rank_config_llama7b.json
    --adalora-rank-budget 2048
    --adalora-allocation-interval 10
    --adalora-diagnostics-path "$NBS_DIAGNOSTICS"
  )
else
  MODEL_TAG="llama_base_low_rank"
  DISPLAY_NAME="NetLLM"
  EXTRA_ARGS=()
fi

TRAIN_PREFIX="his_10_fut_20_ss_15_epochs_${EPOCHS}_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False"
TEST_PREFIX="his_10_fut_20_axes_ss_15_epochs_${EPOCHS}_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False"
MODEL_ROOT="viewport_prediction/data/ft_plms/$MODEL_TAG/freeze_plm_False/multimodal_none/Jin2022/5Hz"
BEST_MODEL="$MODEL_ROOT/$TRAIN_PREFIX/best_model"
RESULT_ROOT="viewport_prediction/data/results/$MODEL_TAG/freeze_plm_False/multimodal_none/Jin2022/5Hz"
RESULT_CSV="$RESULT_ROOT/${TEST_PREFIX}_results.csv"
PARTIAL_RESULT_CSV="$RESULT_ROOT/${TEST_PREFIX}_partial_results.csv"

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

set -e
write_status "training" "running" 0
printf 'variant=%s\nrun_id=%s\nepochs=%s\ncheckpoint_interval=%s\neval_progress_interval=%s\nbest_model=%s\nresult_csv=%s\nnbs_diagnostics=%s\n' \
  "$VARIANT" "$RUN_ID" "$EPOCHS" "$CHECKPOINT_INTERVAL" "$EVAL_PROGRESS_INTERVAL" \
  "$BEST_MODEL" "$RESULT_CSV" "${NBS_DIAGNOSTICS:-}" > "$RUN_DIR/metadata.env"

TRAIN_CMD=(
  python run_plm.py
  --adapt
  --train-dataset Jin2022
  --test-dataset Jin2022
  --plm-type llama
  --plm-size base
  --device cuda
  --device-out cuda
  --fp16
  --gradient-checkpointing
  --rank 32
  --epochs "$EPOCHS"
  --bs 1
  --grad-accum-steps 32
  --steps-per-valid "$CHECKPOINT_INTERVAL"
  --save-checkpoint-per-step "$CHECKPOINT_INTERVAL"
  --save-checkpoint-per-epoch 1
  --seed 1
  "${EXTRA_ARGS[@]}"
)

echo "[$DISPLAY_NAME] training started; artifacts: $RUN_DIR"
if run_logged "$RUN_DIR/train.log" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${TRAIN_CMD[@]}"; then
  :
else
  code=$?
  write_status "training" "failed" "$code"
  echo "Training failed. Existing logs and checkpoints were preserved in $RUN_DIR and $MODEL_ROOT"
  exit "$code"
fi

if [[ ! -d "$BEST_MODEL" ]]; then
  write_status "training" "failed_missing_best_model" 3
  echo "Training exited but best_model was not found: $BEST_MODEL"
  exit 3
fi
write_status "evaluation" "running" 0
touch "$RUN_DIR/evaluation.started"

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
  --rank 32
  --model-path "$BEST_MODEL"
  --epochs "$EPOCHS"
  --bs 1
  --grad-accum-steps 32
  --save-test-progress-per-steps "$EVAL_PROGRESS_INTERVAL"
  --seed 1
  "${EXTRA_ARGS[@]}"
)

echo "[$DISPLAY_NAME] evaluation started"
if run_logged "$RUN_DIR/test.log" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${TEST_CMD[@]}"; then
  :
else
  code=$?
  if [[ -f "$PARTIAL_RESULT_CSV" && "$PARTIAL_RESULT_CSV" -nt "$RUN_DIR/evaluation.started" ]]; then
    cp "$PARTIAL_RESULT_CSV" "$RUN_DIR/partial_results.csv"
  fi
  write_status "evaluation" "failed" "$code"
  echo "Evaluation failed. Completed training outputs were preserved."
  exit "$code"
fi

if [[ ! -f "$RESULT_CSV" || ! "$RESULT_CSV" -nt "$RUN_DIR/evaluation.started" ]]; then
  write_status "evaluation" "failed_missing_results" 4
  echo "Evaluation exited but no newly written result CSV was found: $RESULT_CSV"
  exit 4
fi

cp "$RESULT_CSV" "$RUN_DIR/results.csv"
PREDICTIONS_FILE="${RESULT_CSV/_results.csv/_predictions.txt}"
PER_SAMPLE_FILE="${RESULT_CSV/_results.csv/_per_sample_results.csv}"
[[ -f "$PREDICTIONS_FILE" ]] && cp "$PREDICTIONS_FILE" "$RUN_DIR/predictions.txt"
[[ -f "$PER_SAMPLE_FILE" ]] && cp "$PER_SAMPLE_FILE" "$RUN_DIR/per_sample_results.csv"
if [[ -f "$PARTIAL_RESULT_CSV" && "$PARTIAL_RESULT_CSV" -nt "$RUN_DIR/evaluation.started" ]]; then
  cp "$PARTIAL_RESULT_CSV" "$RUN_DIR/partial_results.csv"
fi

PLOT_CMD=(
  python analysis/plot_netllm_experiment.py
  --variant "$VARIANT"
  --train-log "$RUN_DIR/train.log"
  --result-csv "$RUN_DIR/results.csv"
  --output-dir "$RUN_DIR/figures"
)
if [[ "$VARIANT" == "nbs" ]]; then
  PLOT_CMD+=(--allocator-state "$BEST_MODEL/nash_rank_allocator.pt")
  PLOT_CMD+=(--allocator-diagnostics "$NBS_DIAGNOSTICS")
fi

write_status "visualization" "running" 0
if run_logged "$RUN_DIR/plot.log" "${PLOT_CMD[@]}"; then
  :
else
  code=$?
  write_status "visualization" "failed" "$code"
  echo "Visualization failed, but training/evaluation outputs were preserved."
  exit "$code"
fi

write_status "complete" "complete" 0
echo "[$DISPLAY_NAME] complete: $RUN_DIR"
