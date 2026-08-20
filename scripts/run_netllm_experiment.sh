#!/usr/bin/env bash
set -u
set -o pipefail

VARIANT="${1:-}"
if [[ "$VARIANT" != "nbs" && "$VARIANT" != "nbs_v2" && \
      "$VARIANT" != "nbs_v3" && "$VARIANT" != "nbs_v4" && \
      "$VARIANT" != "nbs_v5" && "$VARIANT" != "nbs_v6" && \
      "$VARIANT" != "nbs_v7" && "$VARIANT" != "nbs_v8" && \
      "$VARIANT" != "nbs_v9" && "$VARIANT" != "nbs_v10" && \
      "$VARIANT" != "nbs_v11" && "$VARIANT" != "nbs_v12" && \
      "$VARIANT" != "nbs_v12_repeat" && "$VARIANT" != "nbs_v13" && \
      "$VARIANT" != "nbs_v14" && "$VARIANT" != "nbs_v15" && \
      "$VARIANT" != "nbs_v16" && "$VARIANT" != "nbs_v17" && \
      "$VARIANT" != "uniform_r12" && "$VARIANT" != "uniform_b736" && \
      "$VARIANT" != "adalora_peft_r12" && \
      "$VARIANT" != "plain" ]]; then
  echo "Usage: bash scripts/run_netllm_experiment.sh {nbs|nbs_v2|nbs_v3|nbs_v4|nbs_v5|nbs_v6|nbs_v7|nbs_v8|nbs_v9|nbs_v10|nbs_v11|nbs_v12|nbs_v12_repeat|nbs_v13|nbs_v14|nbs_v15|nbs_v16|nbs_v17|uniform_r12|uniform_b736|adalora_peft_r12|plain}"
  exit 2
fi

EPOCHS="${EPOCHS:-40}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-500}"
EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-500}"
SAVE_PERIODIC_CHECKPOINTS="${SAVE_PERIODIC_CHECKPOINTS:-0}"
LATENCY_WARMUP_STEPS="${LATENCY_WARMUP_STEPS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
ADALORA_EMA_BETA="${ADALORA_EMA_BETA:-0.9}"
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$CHECKPOINT_INTERVAL" =~ ^[1-9][0-9]*$ && \
        "$VALIDATION_INTERVAL" =~ ^[1-9][0-9]*$ && "$EVAL_PROGRESS_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS, CHECKPOINT_INTERVAL, VALIDATION_INTERVAL, and EVAL_PROGRESS_INTERVAL must be positive integers."
  exit 2
fi
if ! [[ "$LATENCY_WARMUP_STEPS" =~ ^[0-9]+$ ]]; then
  echo "LATENCY_WARMUP_STEPS must be a non-negative integer."
  exit 2
fi
if [[ "$SAVE_PERIODIC_CHECKPOINTS" != "0" && "$SAVE_PERIODIC_CHECKPOINTS" != "1" ]]; then
  echo "SAVE_PERIODIC_CHECKPOINTS must be 0 or 1."
  exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_ROOT="viewport_prediction/data/experiment_runs/netllm_vs_nbs"
RUN_DIR="$ARTIFACT_ROOT/$VARIANT/$RUN_ID"
mkdir -p "$RUN_DIR/figures"
printf '%s\n' "$RUN_DIR" > "$ARTIFACT_ROOT/${VARIANT}_latest.txt"

SCHEDULED_SAMPLING="False"
RANK=32
MULTIMODAL_MODE="none"
MODE_ARGS=()
INFERENCE_ARGS=()
INFERENCE_SUFFIX=""
BEST_MODEL_NAME="best_model"
ADALORA_ALLOCATOR_MODE="none"
if [[ "$VARIANT" == "nbs" || "$VARIANT" == "nbs_v2" || \
      "$VARIANT" == "nbs_v3" || "$VARIANT" == "nbs_v4" || \
      "$VARIANT" == "nbs_v5" || "$VARIANT" == "nbs_v6" || \
      "$VARIANT" == "nbs_v7" || "$VARIANT" == "nbs_v8" || \
      "$VARIANT" == "nbs_v9" || "$VARIANT" == "nbs_v10" || \
      "$VARIANT" == "nbs_v11" || "$VARIANT" == "nbs_v12" || \
      "$VARIANT" == "nbs_v12_repeat" || "$VARIANT" == "nbs_v13" || \
      "$VARIANT" == "nbs_v14" || "$VARIANT" == "nbs_v15" || \
      "$VARIANT" == "nbs_v16" || "$VARIANT" == "nbs_v17" ]]; then
  MODEL_TAG="llama_base_low_rank_adalora"
  ADALORA_ALLOCATOR_MODE="nbs"
  DISPLAY_NAME="NBS-NetLLM"
  RANK_CONFIG="configs/adalora_rank_config_llama7b.json"
  RANK_BUDGET=2048
  EXPERIMENT_ARGS=()
  if [[ "$VARIANT" == "nbs_v2" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v2"
    DISPLAY_NAME="NBS-NetLLM v2"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_nbs_v2.json"
    EXPERIMENT_ARGS=(--experiment-tag nbs_v2)
  elif [[ "$VARIANT" == "nbs_v3" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v3"
    DISPLAY_NAME="NBS-NetLLM v3"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_nbs_v3.json"
    RANK_BUDGET=1536
    EXPERIMENT_ARGS=(--experiment-tag nbs_v3)
  elif [[ "$VARIANT" == "nbs_v4" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v4"
    DISPLAY_NAME="NBS-NetLLM v4"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_nbs_v3.json"
    RANK_BUDGET=1536
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v4
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v5" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v5"
    DISPLAY_NAME="NBS-NetLLM v5"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_nbs_v3.json"
    RANK_BUDGET=1536
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    MIX_RATE=0.1
    SCHEDULED_SAMPLING="True"
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v5
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
      --scheduled-sampling
      --mix-rate "$MIX_RATE"
    )
  elif [[ "$VARIANT" == "nbs_v6" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v6"
    DISPLAY_NAME="NBS-NetLLM v6 (min2-max32-budget256, initial-rank4)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min2_max32.json"
    RANK_BUDGET=256
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v6
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v7" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v7"
    DISPLAY_NAME="NBS-NetLLM v7 (min4-max32-budget512, initial-rank8)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=512
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v7
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v8" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v8"
    DISPLAY_NAME="NBS-NetLLM v8 (min4-max32-budget768, initial-rank12)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=768
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v8
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v9" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v9"
    DISPLAY_NAME="NBS-NetLLM v9 (min4-max32-budget896, initial-rank14)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=896
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v9
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v10" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v10"
    DISPLAY_NAME="NBS-NetLLM v10 (min4-max32-budget640, initial-rank10)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=640
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v10
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v11" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v11"
    DISPLAY_NAME="NBS-NetLLM v11 (min2-max32-budget768, initial-rank12)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min2_max32.json"
    RANK_BUDGET=768
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v11
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v12" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v12"
    DISPLAY_NAME="NBS-NetLLM v12 (min4-max32-budget736, initial-mean-rank11.5)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=736
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v12
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v12_repeat" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v12_repeat"
    DISPLAY_NAME="NBS-NetLLM v12 repeat (min4-max32-budget736, seed1)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=736
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v12_repeat
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v13" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v13"
    DISPLAY_NAME="NBS-NetLLM v13 (min4-max32-budget720, initial-mean-rank11.25, seed1)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=720
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v13
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v14" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v14"
    DISPLAY_NAME="NBS-NetLLM v14 (min4-max32-budget736, lr1.5e-4, ema0.9, seed1)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=736
    LEARNING_RATE=0.00015
    ADALORA_EMA_BETA=0.9
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v14
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v15" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v15"
    DISPLAY_NAME="NBS-NetLLM v15 (min4-max32-budget736, lr2e-4, ema0.95, seed1)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=736
    LEARNING_RATE=0.0002
    ADALORA_EMA_BETA=0.95
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v15
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v16" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v16"
    DISPLAY_NAME="NBS-NetLLM v16 (min4-max32-budget736, lr2.5e-4, ema0.9, seed1)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=736
    LEARNING_RATE=0.00025
    ADALORA_EMA_BETA=0.9
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v16
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  elif [[ "$VARIANT" == "nbs_v17" ]]; then
    MODEL_TAG="llama_base_low_rank_adalora_nbs_v17"
    DISPLAY_NAME="NBS-NetLLM v17 (min4-max32-budget736, lr2e-4, ema0.8, seed1)"
    RANK_CONFIG="configs/adalora_rank_config_llama7b_min4_max32.json"
    RANK_BUDGET=736
    LEARNING_RATE=0.0002
    ADALORA_EMA_BETA=0.8
    EARLY_STOPPING_PATIENCE=2
    EARLY_STOPPING_MIN_DELTA=0.0001
    EXPERIMENT_ARGS=(
      --experiment-tag nbs_v17
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
      --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
    )
  fi
  NBS_DIAGNOSTICS="$RUN_DIR/nbs_rank_diagnostics.csv"
  EXTRA_ARGS=(
    --use-adalora
    --adalora-allocator nbs
    --adalora-rank-config "$RANK_CONFIG"
    --adalora-rank-budget "$RANK_BUDGET"
    --adalora-ema-beta "$ADALORA_EMA_BETA"
    --adalora-allocation-interval 10
    --adalora-diagnostics-path "$NBS_DIAGNOSTICS"
    "${EXPERIMENT_ARGS[@]}"
  )
elif [[ "$VARIANT" == "uniform_r12" ]]; then
  MODEL_TAG="llama_base_low_rank_uniform_r12"
  DISPLAY_NAME="Uniform-rank NetLLM (rank12, total active rank768)"
  RANK=12
  RANK_BUDGET=768
  EXTRA_ARGS=(--experiment-tag uniform_r12)
elif [[ "$VARIANT" == "uniform_b736" ]]; then
  MODEL_TAG="llama_base_low_rank_uniform_b736"
  DISPLAY_NAME="Fixed near-uniform NetLLM (32x rank11 + 32x rank12, budget736, seed1)"
  RANK=12
  RANK_BUDGET=736
  LORA_RANK_CONFIG="configs/lora_rank_pattern_llama7b_budget736.json"
  EXTRA_ARGS=(
    --experiment-tag uniform_b736
    --lora-rank-config "$LORA_RANK_CONFIG"
  )
elif [[ "$VARIANT" == "adalora_peft_r12" ]]; then
  MODEL_TAG="llama_base_low_rank_adalora_adalora_peft_r12"
  ADALORA_ALLOCATOR_MODE="peft"
  DISPLAY_NAME="Stock PEFT AdaLoRA (target-rank12) + Selector + Speculative"
  RANK=12
  RANK_BUDGET=768
  BEST_MODEL_NAME="best_ar_model"
  EARLY_STOPPING_PATIENCE=2
  EARLY_STOPPING_MIN_DELTA=0.0001
  SELECTOR_RECENT_K_VALUE="${SELECTOR_RECENT_K:-6}"
  SPECULATIVE_GAMMA_VALUE="${SPECULATIVE_GAMMA:-4}"
  SPECULATIVE_THRESHOLD_VALUE="${SPECULATIVE_THRESHOLD:-0.3}"
  INFERENCE_ARGS=(
    --inference-tag full_stack
    --selector-recent-k "$SELECTOR_RECENT_K_VALUE"
    --speculative-gamma "$SPECULATIVE_GAMMA_VALUE"
    --speculative-threshold "$SPECULATIVE_THRESHOLD_VALUE"
  )
  INFERENCE_SUFFIX="_inference_full_stack"
  EXTRA_ARGS=(
    --use-adalora
    --adalora-allocator peft
    --adalora-allocation-interval 10
    --experiment-tag adalora_peft_r12
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
  )
else
  MODEL_TAG="llama_base_low_rank"
  DISPLAY_NAME="NetLLM"
  EXTRA_ARGS=()
fi

TRAIN_PREFIX="his_10_fut_20_ss_15_epochs_${EPOCHS}_bs_32_lr_${LEARNING_RATE}_seed_1_rank_${RANK}_scheduled_sampling_${SCHEDULED_SAMPLING}"
TEST_PREFIX="his_10_fut_20_axes_ss_15_epochs_${EPOCHS}_bs_32_lr_${LEARNING_RATE}_seed_1_rank_${RANK}_scheduled_sampling_${SCHEDULED_SAMPLING}"
MODEL_ROOT="viewport_prediction/data/ft_plms/$MODEL_TAG/freeze_plm_False/multimodal_${MULTIMODAL_MODE}/Jin2022/5Hz/$RUN_ID"
if [[ -n "${NBS_DIAGNOSTICS:-}" ]]; then
  BEST_MODEL="$MODEL_ROOT/$TRAIN_PREFIX/best_ar_model"
  BEST_POST_NBS_MODEL="$MODEL_ROOT/$TRAIN_PREFIX/best_post_nbs_model"
  FINAL_NBS_MODEL="$MODEL_ROOT/$TRAIN_PREFIX/final_nbs_model"
else
  BEST_MODEL="$MODEL_ROOT/$TRAIN_PREFIX/$BEST_MODEL_NAME"
  BEST_POST_NBS_MODEL=""
  FINAL_NBS_MODEL=""
fi
RESULT_ROOT="viewport_prediction/data/results/$MODEL_TAG/freeze_plm_False/multimodal_${MULTIMODAL_MODE}/Jin2022/5Hz/$RUN_ID"
if [[ -n "${NBS_DIAGNOSTICS:-}" ]]; then
  RESULT_CSV="$RESULT_ROOT/${TEST_PREFIX}_checkpoint_best_ar${INFERENCE_SUFFIX}_results.csv"
  PARTIAL_RESULT_CSV="$RESULT_ROOT/${TEST_PREFIX}_checkpoint_best_ar${INFERENCE_SUFFIX}_partial_results.csv"
else
  RESULT_CSV="$RESULT_ROOT/${TEST_PREFIX}${INFERENCE_SUFFIX}_results.csv"
  PARTIAL_RESULT_CSV="$RESULT_ROOT/${TEST_PREFIX}${INFERENCE_SUFFIX}_partial_results.csv"
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

checkpoint_complete() {
  local checkpoint_path="$1"
  local canonical_path
  canonical_path="$(resolve_checkpoint "$checkpoint_path")" || return 1
  [[ -f "$canonical_path/adapter_model.bin" && \
     -f "$canonical_path/modules_except_plm.bin" && \
     -f "$canonical_path/nash_rank_allocator.pt" && \
     -f "$checkpoint_path/checkpoint_metadata.json" ]]
}

adapter_checkpoint_complete() {
  local checkpoint_path="$1"
  local canonical_path
  canonical_path="$(resolve_checkpoint "$checkpoint_path")" || return 1
  [[ -f "$canonical_path/adapter_model.bin" && \
     -f "$canonical_path/modules_except_plm.bin" && \
     -f "$checkpoint_path/checkpoint_metadata.json" ]]
}

resolve_checkpoint() {
  python analysis/resolve_checkpoint_alias.py "$1"
}

set -e
write_status "training" "running" 0
printf 'variant=%s\nrun_id=%s\nseed=1\nepochs=%s\nvalidation_interval=%s\ncheckpoint_interval=%s\nsave_periodic_checkpoints=%s\neval_progress_interval=%s\nlatency_warmup_steps=%s\nrank=%s\nlearning_rate=%s\nadalora_ema_beta=%s\nadalora_allocator=%s\nmultimodal_mode=%s\npatch_selection_weights=%s\npatch_top_k=%s\nselector_recent_k=%s\nspeculative_gamma=%s\nspeculative_threshold=%s\nbest_ar_model=%s\nbest_post_nbs_model=%s\nfinal_nbs_model=%s\nresult_csv=%s\nnbs_diagnostics=%s\nrank_config=%s\nlora_rank_config=%s\nrank_budget=%s\nearly_stopping_patience=%s\nearly_stopping_min_delta=%s\nscheduled_sampling=%s\nmix_rate=%s\n' \
  "$VARIANT" "$RUN_ID" "$EPOCHS" "$VALIDATION_INTERVAL" "$CHECKPOINT_INTERVAL" \
  "$SAVE_PERIODIC_CHECKPOINTS" "$EVAL_PROGRESS_INTERVAL" "$LATENCY_WARMUP_STEPS" "$RANK" \
  "$LEARNING_RATE" "$ADALORA_EMA_BETA" "$ADALORA_ALLOCATOR_MODE" "$MULTIMODAL_MODE" \
  "${PATCH_SELECTION_WEIGHTS:-}" "${PATCH_TOP_K:-}" "${SELECTOR_RECENT_K_VALUE:-}" \
  "${SPECULATIVE_GAMMA_VALUE:-}" "${SPECULATIVE_THRESHOLD_VALUE:-}" \
  "$BEST_MODEL" "$BEST_POST_NBS_MODEL" "$FINAL_NBS_MODEL" "$RESULT_CSV" \
  "${NBS_DIAGNOSTICS:-}" "${RANK_CONFIG:-}" "${LORA_RANK_CONFIG:-}" \
  "${RANK_BUDGET:-}" "${EARLY_STOPPING_PATIENCE:-}" "${EARLY_STOPPING_MIN_DELTA:-}" \
  "$SCHEDULED_SAMPLING" "${MIX_RATE:-}" > "$RUN_DIR/metadata.env"

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
  --rank "$RANK"
  --experiment-run-id "$RUN_ID"
  --epochs "$EPOCHS"
  --bs 1
  --grad-accum-steps 32
  --steps-per-valid "$VALIDATION_INTERVAL"
  --lr "$LEARNING_RATE"
  --seed 1
  "${MODE_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)
if [[ "$SAVE_PERIODIC_CHECKPOINTS" == "1" ]]; then
  TRAIN_CMD+=(
    --save-checkpoint-per-step "$CHECKPOINT_INTERVAL"
    --save-checkpoint-per-epoch 1
  )
fi

echo "[$DISPLAY_NAME] training started; artifacts: $RUN_DIR"
if run_logged "$RUN_DIR/train.log" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${TRAIN_CMD[@]}"; then
  :
else
  code=$?
  write_status "training" "failed" "$code"
  echo "Training failed. Existing logs and checkpoints were preserved in $RUN_DIR and $MODEL_ROOT"
  exit "$code"
fi

if [[ -n "${NBS_DIAGNOSTICS:-}" ]] && ! checkpoint_complete "$BEST_MODEL"; then
  write_status "training" "failed_missing_best_model" 3
  echo "Training exited but the best AR checkpoint is incomplete: $BEST_MODEL"
  exit 3
elif [[ -z "${NBS_DIAGNOSTICS:-}" ]] && ! adapter_checkpoint_complete "$BEST_MODEL"; then
  write_status "training" "failed_missing_best_model" 3
  echo "Training exited but best_model was not found: $BEST_MODEL"
  exit 3
fi
if [[ -n "${NBS_DIAGNOSTICS:-}" ]]; then
  CHECKPOINT_ROLES=(best_ar best_post_nbs final_nbs)
  CHECKPOINT_PATHS=("$BEST_MODEL" "$BEST_POST_NBS_MODEL" "$FINAL_NBS_MODEL")
else
  CHECKPOINT_ROLES=(best)
  CHECKPOINT_PATHS=("$BEST_MODEL")
fi
declare -A EVALUATED_DIR_BY_CHECKPOINT=()
declare -A EVALUATED_ROLE_BY_CHECKPOINT=()

for index in "${!CHECKPOINT_ROLES[@]}"; do
  ROLE="${CHECKPOINT_ROLES[$index]}"
  MODEL_PATH="${CHECKPOINT_PATHS[$index]}"
  if [[ -n "${NBS_DIAGNOSTICS:-}" ]] && ! checkpoint_complete "$MODEL_PATH"; then
    if [[ "$ROLE" == "best_post_nbs" ]]; then
      mkdir -p "$RUN_DIR/evaluations/$ROLE"
      printf 'No successful NBS allocation occurred before validation; checkpoint unavailable.\n' \
        > "$RUN_DIR/evaluations/$ROLE/skipped.txt"
      echo "[$DISPLAY_NAME] skipping $ROLE evaluation: checkpoint unavailable"
      continue
    fi
    write_status "evaluation_${ROLE}" "failed_missing_checkpoint" 3
    echo "Required checkpoint was not found: $MODEL_PATH"
    exit 3
  elif [[ -z "${NBS_DIAGNOSTICS:-}" ]] && ! adapter_checkpoint_complete "$MODEL_PATH"; then
    write_status "evaluation_${ROLE}" "failed_missing_checkpoint" 3
    echo "Required adapter checkpoint is incomplete: $MODEL_PATH"
    exit 3
  fi

  EVALUATION_TAG_ARGS=()
  if [[ "$ROLE" == "best" ]]; then
    ROLE_DIR="$RUN_DIR"
    ROLE_RESULT_CSV="$RESULT_CSV"
    ROLE_PARTIAL_RESULT_CSV="$PARTIAL_RESULT_CSV"
  else
    ROLE_DIR="$RUN_DIR/evaluations/$ROLE"
    ROLE_RESULT_CSV="$RESULT_ROOT/${TEST_PREFIX}_checkpoint_${ROLE}${INFERENCE_SUFFIX}_results.csv"
    ROLE_PARTIAL_RESULT_CSV="$RESULT_ROOT/${TEST_PREFIX}_checkpoint_${ROLE}${INFERENCE_SUFFIX}_partial_results.csv"
    EVALUATION_TAG_ARGS=(--evaluation-tag "$ROLE")
  fi
  CANONICAL_MODEL_PATH="$(resolve_checkpoint "$MODEL_PATH")"
  mkdir -p "$ROLE_DIR/figures"
  PREDICTIONS_FILE="${ROLE_RESULT_CSV/_results.csv/_predictions.txt}"
  PER_SAMPLE_FILE="${ROLE_RESULT_CSV/_results.csv/_per_sample_results.csv}"
  LATENCY_FILE="${ROLE_RESULT_CSV/_results.csv/_latency.json}"
  LATENCY_DETAIL_FILE="${LATENCY_FILE%.json}_per_sample.csv"
  LATENCY_PARTIAL_FILE="${LATENCY_FILE%.json}_partial.json"
  LATENCY_PARTIAL_DETAIL_FILE="${LATENCY_FILE%.json}_partial_per_sample.csv"
  INFERENCE_TRACE_FILE="${ROLE_RESULT_CSV/_results.csv/_inference_trace.json}"
  INFERENCE_TRACE_DETAIL_FILE="${INFERENCE_TRACE_FILE%.json}_per_sample.csv"
  REUSED_FROM_DIR="${EVALUATED_DIR_BY_CHECKPOINT[$CANONICAL_MODEL_PATH]:-}"
  REUSED_FROM_ROLE="${EVALUATED_ROLE_BY_CHECKPOINT[$CANONICAL_MODEL_PATH]:-}"

  if [[ -n "$REUSED_FROM_DIR" ]]; then
    write_status "evaluation_${ROLE}" "reused" 0
    mkdir -p "$(dirname "$ROLE_RESULT_CSV")"
    cp "$REUSED_FROM_DIR/results.csv" "$ROLE_RESULT_CSV"
    cp "$REUSED_FROM_DIR/results.csv" "$ROLE_DIR/results.csv"
    if [[ -f "$REUSED_FROM_DIR/predictions.txt" ]]; then
      cp "$REUSED_FROM_DIR/predictions.txt" "$PREDICTIONS_FILE"
      cp "$REUSED_FROM_DIR/predictions.txt" "$ROLE_DIR/predictions.txt"
    fi
    if [[ -f "$REUSED_FROM_DIR/per_sample_results.csv" ]]; then
      cp "$REUSED_FROM_DIR/per_sample_results.csv" "$PER_SAMPLE_FILE"
      cp "$REUSED_FROM_DIR/per_sample_results.csv" "$ROLE_DIR/per_sample_results.csv"
    fi
    if [[ -f "$REUSED_FROM_DIR/partial_results.csv" ]]; then
      cp "$REUSED_FROM_DIR/partial_results.csv" "$ROLE_PARTIAL_RESULT_CSV"
      cp "$REUSED_FROM_DIR/partial_results.csv" "$ROLE_DIR/partial_results.csv"
    fi
    if [[ -f "$REUSED_FROM_DIR/latency.json" ]]; then
      cp "$REUSED_FROM_DIR/latency.json" "$LATENCY_FILE"
      cp "$REUSED_FROM_DIR/latency.json" "$ROLE_DIR/latency.json"
    fi
    if [[ -f "$REUSED_FROM_DIR/latency_per_sample.csv" ]]; then
      cp "$REUSED_FROM_DIR/latency_per_sample.csv" "$LATENCY_DETAIL_FILE"
      cp "$REUSED_FROM_DIR/latency_per_sample.csv" "$ROLE_DIR/latency_per_sample.csv"
    fi
    if [[ -f "$REUSED_FROM_DIR/inference_trace.json" ]]; then
      cp "$REUSED_FROM_DIR/inference_trace.json" "$INFERENCE_TRACE_FILE"
      cp "$REUSED_FROM_DIR/inference_trace.json" "$ROLE_DIR/inference_trace.json"
    fi
    if [[ -f "$REUSED_FROM_DIR/inference_trace_per_sample.csv" ]]; then
      cp "$REUSED_FROM_DIR/inference_trace_per_sample.csv" "$INFERENCE_TRACE_DETAIL_FILE"
      cp "$REUSED_FROM_DIR/inference_trace_per_sample.csv" "$ROLE_DIR/inference_trace_per_sample.csv"
    fi
    printf '{\n  "checkpoint_role": "%s",\n  "reused_from_role": "%s",\n  "canonical_checkpoint": "%s"\n}\n' \
      "$ROLE" "$REUSED_FROM_ROLE" "$CANONICAL_MODEL_PATH" \
      > "$ROLE_DIR/evaluation_reused.json"
    printf 'Evaluation reused from %s because both roles resolve to %s\n' \
      "$REUSED_FROM_ROLE" "$CANONICAL_MODEL_PATH" > "$ROLE_DIR/test.log"
    echo "[$DISPLAY_NAME] $ROLE evaluation reused from $REUSED_FROM_ROLE"
  else
    touch "$ROLE_DIR/evaluation.started"
    write_status "evaluation_${ROLE}" "running" 0

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
      --model-path "$MODEL_PATH"
      "${EVALUATION_TAG_ARGS[@]}"
      --epochs "$EPOCHS"
      --bs 1
      --grad-accum-steps 32
      --lr "$LEARNING_RATE"
      --save-test-progress-per-steps "$EVAL_PROGRESS_INTERVAL"
      --measure-inference-latency
      --latency-warmup-steps "$LATENCY_WARMUP_STEPS"
      --latency-output-path "$LATENCY_FILE"
      --seed 1
      "${MODE_ARGS[@]}"
      "${INFERENCE_ARGS[@]}"
      "${EXTRA_ARGS[@]}"
    )

    echo "[$DISPLAY_NAME] $ROLE evaluation started"
    if run_logged "$ROLE_DIR/test.log" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${TEST_CMD[@]}"; then
      :
    else
      code=$?
      if [[ -f "$ROLE_PARTIAL_RESULT_CSV" && \
            "$ROLE_PARTIAL_RESULT_CSV" -nt "$ROLE_DIR/evaluation.started" ]]; then
        cp "$ROLE_PARTIAL_RESULT_CSV" "$ROLE_DIR/partial_results.csv"
      fi
      [[ -f "$LATENCY_PARTIAL_FILE" ]] && cp "$LATENCY_PARTIAL_FILE" "$ROLE_DIR/latency_partial.json"
      [[ -f "$LATENCY_PARTIAL_DETAIL_FILE" ]] && cp "$LATENCY_PARTIAL_DETAIL_FILE" "$ROLE_DIR/latency_partial_per_sample.csv"
      write_status "evaluation_${ROLE}" "failed" "$code"
      echo "$ROLE evaluation failed. Completed outputs were preserved."
      exit "$code"
    fi

    if [[ ! -f "$ROLE_RESULT_CSV" || \
          ! "$ROLE_RESULT_CSV" -nt "$ROLE_DIR/evaluation.started" ]]; then
      write_status "evaluation_${ROLE}" "failed_missing_results" 4
      echo "Evaluation exited but no newly written result CSV was found: $ROLE_RESULT_CSV"
      exit 4
    fi

    cp "$ROLE_RESULT_CSV" "$ROLE_DIR/results.csv"
    [[ -f "$PREDICTIONS_FILE" ]] && cp "$PREDICTIONS_FILE" "$ROLE_DIR/predictions.txt"
    [[ -f "$PER_SAMPLE_FILE" ]] && cp "$PER_SAMPLE_FILE" "$ROLE_DIR/per_sample_results.csv"
    [[ -f "$LATENCY_FILE" ]] && cp "$LATENCY_FILE" "$ROLE_DIR/latency.json"
    [[ -f "$LATENCY_DETAIL_FILE" ]] && cp "$LATENCY_DETAIL_FILE" "$ROLE_DIR/latency_per_sample.csv"
    [[ -f "$INFERENCE_TRACE_FILE" ]] && cp "$INFERENCE_TRACE_FILE" "$ROLE_DIR/inference_trace.json"
    [[ -f "$INFERENCE_TRACE_DETAIL_FILE" ]] && cp "$INFERENCE_TRACE_DETAIL_FILE" "$ROLE_DIR/inference_trace_per_sample.csv"
    if [[ -f "$ROLE_PARTIAL_RESULT_CSV" && \
          "$ROLE_PARTIAL_RESULT_CSV" -nt "$ROLE_DIR/evaluation.started" ]]; then
      cp "$ROLE_PARTIAL_RESULT_CSV" "$ROLE_DIR/partial_results.csv"
    fi
    EVALUATED_DIR_BY_CHECKPOINT["$CANONICAL_MODEL_PATH"]="$ROLE_DIR"
    EVALUATED_ROLE_BY_CHECKPOINT["$CANONICAL_MODEL_PATH"]="$ROLE"
  fi

  if [[ ! -f "$ROLE_DIR/latency.json" ]]; then
    write_status "evaluation_${ROLE}" "failed_missing_latency" 5
    echo "Evaluation completed but latency summary is missing: $ROLE_DIR/latency.json"
    exit 5
  fi
  if [[ -n "$INFERENCE_SUFFIX" && ! -f "$ROLE_DIR/inference_trace.json" ]]; then
    write_status "evaluation_${ROLE}" "failed_missing_inference_trace" 6
    echo "Evaluation completed but inference trace is missing: $ROLE_DIR/inference_trace.json"
    exit 6
  fi

  PLOT_CMD=(
    python analysis/plot_netllm_experiment.py
    --variant "$VARIANT"
    --train-log "$RUN_DIR/train.log"
    --result-csv "$ROLE_DIR/results.csv"
    --output-dir "$ROLE_DIR/figures"
    --checkpoint-role "$ROLE"
    --latency-json "$ROLE_DIR/latency.json"
  )
  if [[ -n "${NBS_DIAGNOSTICS:-}" ]]; then
    PLOT_CMD+=(--allocator-state "$CANONICAL_MODEL_PATH/nash_rank_allocator.pt")
    PLOT_CMD+=(--allocator-diagnostics "$NBS_DIAGNOSTICS")
  fi

  write_status "visualization_${ROLE}" "running" 0
  if run_logged "$ROLE_DIR/plot.log" "${PLOT_CMD[@]}"; then
    :
  else
    code=$?
    write_status "visualization_${ROLE}" "failed" "$code"
    echo "$ROLE visualization failed, but training/evaluation outputs were preserved."
    exit "$code"
  fi

  # Preserve the historical top-level files as aliases of the primary result.
  if [[ "$ROLE" == "best_ar" ]]; then
    cp "$ROLE_DIR/results.csv" "$RUN_DIR/results.csv"
    [[ -f "$ROLE_DIR/predictions.txt" ]] && cp "$ROLE_DIR/predictions.txt" "$RUN_DIR/predictions.txt"
    [[ -f "$ROLE_DIR/per_sample_results.csv" ]] && cp "$ROLE_DIR/per_sample_results.csv" "$RUN_DIR/per_sample_results.csv"
    [[ -f "$ROLE_DIR/latency.json" ]] && cp "$ROLE_DIR/latency.json" "$RUN_DIR/latency.json"
    [[ -f "$ROLE_DIR/latency_per_sample.csv" ]] && cp "$ROLE_DIR/latency_per_sample.csv" "$RUN_DIR/latency_per_sample.csv"
    cp -a "$ROLE_DIR/figures/." "$RUN_DIR/figures/"
  fi
done

write_status "complete" "complete" 0
echo "[$DISPLAY_NAME] complete: $RUN_DIR"
