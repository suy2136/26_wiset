#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

V19_CHECKPOINT="${V19_CHECKPOINT:-viewport_prediction/data/ft_plms/llama_base_low_rank_adalora_nbs_v19/freeze_plm_False/multimodal_none/Jin2022/5Hz/20260821_204908/his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False/best_ar_model}"
PROJECTOR_CHECKPOINT="${PROJECTOR_CHECKPOINT:-patch_selection_delivery/model/vp_3condition_best_model}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19_frozen_patch_selector/$RUN_ID}"
SELECTOR_EPOCHS="${SELECTOR_EPOCHS:-30}"
SELECTOR_BATCH_SIZE="${SELECTOR_BATCH_SIZE:-32}"
SELECTOR_LR="${SELECTOR_LR:-0.0001}"
PATCH_TOP_K="${PATCH_TOP_K:-8}"
PATCH_BUDGET_PENALTY="${PATCH_BUDGET_PENALTY:-1.0}"

if [[ ! -f "$V19_CHECKPOINT/adapter_model.bin" || \
      ! -f "$V19_CHECKPOINT/modules_except_plm.bin" || \
      ! -f "$V19_CHECKPOINT/nash_rank_allocator.pt" ]]; then
  echo "Incomplete NBS-v19 checkpoint: $V19_CHECKPOINT" >&2
  exit 2
fi
if [[ -d "$PROJECTOR_CHECKPOINT" ]]; then
  PROJECTOR_FILE="$PROJECTOR_CHECKPOINT/modules_except_plm.bin"
else
  PROJECTOR_FILE="$PROJECTOR_CHECKPOINT"
fi
if [[ ! -f "$PROJECTOR_FILE" ]]; then
  echo "Multimodal projector checkpoint not found: $PROJECTOR_FILE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$OUTPUT_DIR" > \
  viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19_frozen_patch_selector_latest.txt

EXTRA_LIMIT_ARGS=()
if [[ -n "${LIMIT_TRAIN_SAMPLES:-}" ]]; then
  EXTRA_LIMIT_ARGS+=(--limit-train-samples "$LIMIT_TRAIN_SAMPLES")
fi
if [[ -n "${LIMIT_VALID_SAMPLES:-}" ]]; then
  EXTRA_LIMIT_ARGS+=(--limit-valid-samples "$LIMIT_VALID_SAMPLES")
fi
INITIAL_SELECTOR_ARGS=()
if [[ -n "${INITIAL_SELECTOR_WEIGHTS:-}" ]]; then
  INITIAL_SELECTOR_ARGS+=(--patch-selection-weights "$INITIAL_SELECTOR_WEIGHTS")
fi

python run_plm.py \
  --adapt \
  --train-dataset Jin2022 \
  --test-dataset Jin2022 \
  --plm-type llama \
  --plm-size base \
  --device cuda \
  --device-out cuda \
  --fp16 \
  --rank 32 \
  --use-adalora \
  --adalora-allocator nbs \
  --adalora-rank-config configs/adalora_rank_config_llama7b_min2_max32.json \
  --adalora-rank-budget 512 \
  --adalora-ema-beta 0.9 \
  --adalora-shadow-update-policy legacy \
  --adalora-allocation-interval 10 \
  --resume \
  --resume-path "$V19_CHECKPOINT" \
  --multimodal-mode patch-selection \
  --multimodal-projector-checkpoint "$PROJECTOR_FILE" \
  --train-patch-selector-only \
  --selector-objective bce \
  --patch-selector-output-dir "$OUTPUT_DIR" \
  --patch-top-k "$PATCH_TOP_K" \
  --patch-selector-budget-penalty "$PATCH_BUDGET_PENALTY" \
  --epochs "$SELECTOR_EPOCHS" \
  --bs "$SELECTOR_BATCH_SIZE" \
  --grad-accum-steps 1 \
  --lr "$SELECTOR_LR" \
  --seed 1 \
  --lora-seed 1 \
  --data-seed 1 \
  "${EXTRA_LIMIT_ARGS[@]}" \
  "${INITIAL_SELECTOR_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Frozen NBS-v19 selector training complete: $OUTPUT_DIR"
