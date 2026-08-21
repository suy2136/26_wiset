#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NBS_CHECKPOINT="${NBS_CHECKPOINT:-viewport_prediction/data/ft_plms/llama_base_low_rank_adalora_nbs_v12_repeat/freeze_plm_False/multimodal_none/Jin2022/5Hz/20260819_152039/his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False/best_ar_model}"
UNIFORM_CHECKPOINT="${UNIFORM_CHECKPOINT:-viewport_prediction/data/ft_plms/llama_base_low_rank_uniform_b736/freeze_plm_False/multimodal_none/Jin2022/5Hz/20260820_011853/his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_12_scheduled_sampling_False/best_model}"
ADALORA_METADATA="${ADALORA_METADATA:-viewport_prediction/data/experiment_runs/netllm_vs_nbs/adalora_peft_r12/20260820_162747/metadata.env}"

if [[ -z "${ADALORA_CHECKPOINT:-}" ]]; then
  if [[ ! -f "$ADALORA_METADATA" ]]; then
    echo "AdaLoRA metadata missing: $ADALORA_METADATA"
    echo "Set ADALORA_METADATA or ADALORA_CHECKPOINT to the server path."
    exit 2
  fi
  ADALORA_CHECKPOINT="$(sed -n 's/^best_ar_model=//p' "$ADALORA_METADATA")"
fi

adapter_checkpoint_complete() {
  local path="$1"
  [[ -d "$path" ]] && \
    { [[ -f "$path/adapter_model.bin" ]] || [[ -f "$path/adapter_model.safetensors" ]]; } && \
    [[ -f "$path/modules_except_plm.bin" ]]
}

for checkpoint in "$NBS_CHECKPOINT" "$UNIFORM_CHECKPOINT" "$ADALORA_CHECKPOINT"; do
  if ! adapter_checkpoint_complete "$checkpoint"; then
    echo "Incomplete checkpoint: $checkpoint"
    exit 2
  fi
done
if [[ ! -f "$NBS_CHECKPOINT/nash_rank_allocator.pt" ]]; then
  echo "NBS allocator state missing: $NBS_CHECKPOINT/nash_rank_allocator.pt"
  exit 2
fi

python -c "import torch, peft, transformers; print('LLM evaluation dependencies OK')"
python analysis/verify_viewport_datasets.py \
  --datasets Wu2017 --splits test --frequency 5

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="viewport_prediction/data/experiment_runs/unseen_wu2017/$RUN_ID"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > viewport_prediction/data/experiment_runs/unseen_wu2017_latest.txt

COMMON_ARGS=(
  --test
  --train-dataset Jin2022
  --test-dataset Wu2017
  --plm-type llama
  --plm-size base
  --device cuda
  --device-out cuda
  --fp16
  --epochs 4
  --bs 1
  --grad-accum-steps 32
  --lr 0.0002
  --seed 1
  --save-test-progress-per-steps "${EVAL_PROGRESS_INTERVAL:-500}"
  --measure-inference-latency
  --latency-warmup-steps "${LATENCY_WARMUP_STEPS:-5}"
)
NBS_ARGS=(
  --rank 32
  --use-adalora
  --adalora-allocator nbs
  --adalora-rank-config configs/adalora_rank_config_llama7b_min4_max32.json
  --adalora-rank-budget 736
  --experiment-tag nbs_v12_repeat
)
UNIFORM_ARGS=(
  --rank 12
  --lora-rank-config configs/lora_rank_pattern_llama7b_budget736.json
  --experiment-tag uniform_b736
)
ADALORA_ARGS=(
  --rank 12
  --use-adalora
  --adalora-allocator peft
  --experiment-tag adalora_peft_r12
)

run_case() {
  local number="$1"
  local case_name="$2"
  local checkpoint="$3"
  local inference_mode="$4"
  local config_name="$5"
  local output_dir="$RUN_DIR/${number}_${case_name}"
  local -a config_args=()
  local -a inference_args=()

  case "$config_name" in
    nbs) config_args=("${NBS_ARGS[@]}") ;;
    uniform) config_args=("${UNIFORM_ARGS[@]}") ;;
    adalora) config_args=("${ADALORA_ARGS[@]}") ;;
    *) echo "Unknown model configuration: $config_name"; exit 2 ;;
  esac
  case "$inference_mode" in
    direct) ;;
    selector)
      inference_args=(--inference-tag selector --selector-recent-k "${SELECTOR_RECENT_K:-6}")
      ;;
    speculative)
      inference_args=(
        --inference-tag speculative
        --speculative-gamma "${SPECULATIVE_GAMMA:-4}"
        --speculative-threshold "${SPECULATIVE_THRESHOLD:-0.3}"
      )
      ;;
    full_stack)
      inference_args=(
        --inference-tag full_stack
        --selector-recent-k "${SELECTOR_RECENT_K:-6}"
        --speculative-gamma "${SPECULATIVE_GAMMA:-4}"
        --speculative-threshold "${SPECULATIVE_THRESHOLD:-0.3}"
      )
      ;;
    *) echo "Unknown inference mode: $inference_mode"; exit 2 ;;
  esac

  mkdir -p "$output_dir"
  echo "[$number/8] Starting $case_name on unseen Wu2017"
  python run_plm.py \
    "${COMMON_ARGS[@]}" \
    --model-path "$checkpoint" \
    --results-output-dir "$output_dir" \
    --latency-output-path "$output_dir/latency.json" \
    --inference-trace-output-path "$output_dir/inference_trace.json" \
    "${config_args[@]}" \
    "${inference_args[@]}" \
    2>&1 | tee "$output_dir/evaluation.log"

  local result_csv
  result_csv="$(find "$output_dir" -maxdepth 1 -type f -name '*_results.csv' \
    ! -name '*_partial_results.csv' ! -name '*_per_sample_results.csv' | head -1)"
  if [[ -z "$result_csv" ]]; then
    echo "Result CSV missing for $case_name"
    exit 3
  fi
  cp "$result_csv" "$output_dir/results.csv"
  local per_sample="${result_csv/_results.csv/_per_sample_results.csv}"
  [[ -f "$per_sample" ]] && cp "$per_sample" "$output_dir/per_sample_results.csv"
  printf '{"case":"%s","dataset":"Wu2017","status":"complete"}\n' \
    "$case_name" > "$output_dir/status.json"
}

run_case 1 nbs_v12_repeat_direct "$NBS_CHECKPOINT" direct nbs
run_case 2 nbs_v12_repeat_selector "$NBS_CHECKPOINT" selector nbs
run_case 3 nbs_v12_repeat_speculative "$NBS_CHECKPOINT" speculative nbs
run_case 4 nbs_v12_repeat_full_stack "$NBS_CHECKPOINT" full_stack nbs
run_case 5 uniform_b736_direct "$UNIFORM_CHECKPOINT" direct uniform
run_case 6 adalora_direct "$ADALORA_CHECKPOINT" direct adalora
run_case 7 uniform_b736_full_stack "$UNIFORM_CHECKPOINT" full_stack uniform
run_case 8 adalora_full_stack "$ADALORA_CHECKPOINT" full_stack adalora

printf '{"run_id":"%s","dataset":"Wu2017","status":"complete","run_dir":"%s"}\n' \
  "$RUN_ID" "$RUN_DIR" > "$RUN_DIR/status.json"
echo "Eight-model unseen Wu2017 evaluation complete: $RUN_DIR"
