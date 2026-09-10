#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

V19_CHECKPOINT="${V19_CHECKPOINT:-viewport_prediction/data/ft_plms/llama_base_low_rank_adalora_nbs_v19/freeze_plm_False/multimodal_none/Jin2022/5Hz/20260821_204908/his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False/best_ar_model}"
PROJECTOR_CHECKPOINT="${PROJECTOR_CHECKPOINT:-patch_selection_delivery/model/vp_3condition_best_model}"
BCE_SELECTOR_WEIGHTS="${BCE_SELECTOR_WEIGHTS:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19_patch_selector_nbs_prediction/$RUN_ID}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
FINETUNE_LR="${FINETUNE_LR:-0.00002}"
PATCH_TOP_K="${PATCH_TOP_K:-8}"
BCE_WEIGHT="${BCE_WEIGHT:-0.1}"
BUDGET_PENALTY="${BUDGET_PENALTY:-1.0}"
ST_TEMPERATURE="${ST_TEMPERATURE:-1.0}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-128}"

if [[ -z "$BCE_SELECTOR_WEIGHTS" ]]; then
  POINTER="viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19_frozen_patch_selector_latest.txt"
  if [[ ! -f "$POINTER" ]]; then
    echo "Set BCE_SELECTOR_WEIGHTS to stage-1 best_patch_selector.pth." >&2
    exit 2
  fi
  BCE_RUN_DIR="$(<"$POINTER")"
  BCE_SELECTOR_WEIGHTS="$BCE_RUN_DIR/best_patch_selector.pth"
fi
if [[ -d "$PROJECTOR_CHECKPOINT" ]]; then
  PROJECTOR_FILE="$PROJECTOR_CHECKPOINT/modules_except_plm.bin"
else
  PROJECTOR_FILE="$PROJECTOR_CHECKPOINT"
fi
for required in \
  "$V19_CHECKPOINT/adapter_model.bin" \
  "$V19_CHECKPOINT/modules_except_plm.bin" \
  "$V19_CHECKPOINT/nash_rank_allocator.pt" \
  "$PROJECTOR_FILE" \
  "$BCE_SELECTOR_WEIGHTS"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 3
  fi
done
if [[ ! -d viewport_prediction/data/images/Jin2022_images ]]; then
  echo "Raw Jin2022 frames not found." >&2
  exit 3
fi

mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$OUTPUT_DIR" > \
  viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19_patch_selector_nbs_prediction_latest.txt

EXTRA_LIMIT_ARGS=()
if [[ -n "${LIMIT_TRAIN_SAMPLES:-}" ]]; then
  EXTRA_LIMIT_ARGS+=(--limit-train-samples "$LIMIT_TRAIN_SAMPLES")
fi
if [[ -n "${LIMIT_VALID_SAMPLES:-}" ]]; then
  EXTRA_LIMIT_ARGS+=(--limit-valid-samples "$LIMIT_VALID_SAMPLES")
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
  --patch-selection-weights "$BCE_SELECTOR_WEIGHTS" \
  --train-patch-selector-only \
  --selector-objective nbs-prediction \
  --patch-selector-output-dir "$OUTPUT_DIR" \
  --patch-top-k "$PATCH_TOP_K" \
  --patch-selector-bce-weight "$BCE_WEIGHT" \
  --patch-selector-budget-penalty "$BUDGET_PENALTY" \
  --patch-selector-st-temperature "$ST_TEMPERATURE" \
  --patch-selector-validation-samples "$VALIDATION_SAMPLES" \
  --epochs "$FINETUNE_EPOCHS" \
  --bs 1 \
  --grad-accum-steps 32 \
  --lr "$FINETUNE_LR" \
  --seed 1 \
  --lora-seed 1 \
  --data-seed 1 \
  "${EXTRA_LIMIT_ARGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/finetune.log"

echo "NBS-prediction selector fine-tuning complete: $OUTPUT_DIR"
