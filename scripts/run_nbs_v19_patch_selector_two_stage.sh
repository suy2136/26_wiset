#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-run}"
if [[ "$MODE" != "run" && "$MODE" != "dry-run" ]]; then
  echo "usage: $0 [run|dry-run]" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${OUTPUT_ROOT:-viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19_patch_selector_two_stage/$RUN_ID}"
V19_RUN_DIR="${V19_RUN_DIR:-viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19/20260821_204908}"
V19_CHECKPOINT="${V19_CHECKPOINT:-viewport_prediction/data/ft_plms/llama_base_low_rank_adalora_nbs_v19/freeze_plm_False/multimodal_none/Jin2022/5Hz/20260821_204908/his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False/best_ar_model}"
BCE_DIR="$ROOT/stage1_bce"
FINETUNE_DIR="$ROOT/stage2_nbs_prediction"
EVAL_DIR="$ROOT/stage3_evaluation"
mkdir -p "$BCE_DIR" "$FINETUNE_DIR" "$EVAL_DIR"

if [[ "$MODE" == "dry-run" ]]; then
  OUTPUT_DIR="$BCE_DIR" LIMIT_TRAIN_SAMPLES=2 LIMIT_VALID_SAMPLES=1 \
    bash scripts/train_nbs_v19_frozen_patch_selector.sh --help >/dev/null 2>&1 || true
  echo "stage1=$BCE_DIR"
  echo "stage2=$FINETUNE_DIR"
  echo "stage3=$EVAL_DIR"
  exit 0
fi

V19_CHECKPOINT="$V19_CHECKPOINT" OUTPUT_DIR="$BCE_DIR" \
  bash scripts/train_nbs_v19_frozen_patch_selector.sh

BCE_SELECTOR_WEIGHTS="$BCE_DIR/best_patch_selector.pth" \
V19_CHECKPOINT="$V19_CHECKPOINT" \
OUTPUT_DIR="$FINETUNE_DIR" \
  bash scripts/finetune_nbs_v19_patch_selector.sh

PATCH_SELECTION_WEIGHTS="$FINETUNE_DIR/best_patch_selector.pth" \
V19_RUN_DIR="$V19_RUN_DIR" \
NBS_CHECKPOINT="$V19_CHECKPOINT" \
MULTIMODAL_PROJECTOR_CHECKPOINT="${PROJECTOR_CHECKPOINT:-patch_selection_delivery/model/vp_3condition_best_model}" \
PATCH_TOP_K="${PATCH_TOP_K:-8}" \
PATCH_SELECTOR_RUN_DIR="$EVAL_DIR" \
  bash scripts/evaluate_nbs_v19_with_patch_selection.sh run

printf '%s\n' "$ROOT" > \
  viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v19_patch_selector_two_stage_latest.txt
echo "Two-stage selector experiment complete: $ROOT"
