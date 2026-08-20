#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NBS_CHECKPOINT="${NBS_CHECKPOINT:-viewport_prediction/data/ft_plms/llama_base_low_rank_adalora_nbs_v12_repeat/freeze_plm_False/multimodal_none/Jin2022/5Hz/20260819_152039/his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_scheduled_sampling_False/best_ar_model}"
UNIFORM_CHECKPOINT="${UNIFORM_CHECKPOINT:-viewport_prediction/data/ft_plms/llama_base_low_rank_uniform_b736/freeze_plm_False/multimodal_none/Jin2022/5Hz/20260820_011853/his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_12_scheduled_sampling_False/best_model}"
NBS_TRAIN_LOG="${NBS_TRAIN_LOG:-viewport_prediction/data/experiment_runs/netllm_vs_nbs/nbs_v12_repeat/20260819_152039/train.log}"
UNIFORM_TRAIN_LOG="${UNIFORM_TRAIN_LOG:-viewport_prediction/data/experiment_runs/netllm_vs_nbs/uniform_b736/20260820_011853/train.log}"

for path in "$NBS_CHECKPOINT" "$UNIFORM_CHECKPOINT"; do
  if [[ ! -f "$path/adapter_model.bin" || ! -f "$path/modules_except_plm.bin" ]]; then
    echo "Incomplete checkpoint: $path"
    exit 2
  fi
done
if [[ ! -f "$NBS_CHECKPOINT/nash_rank_allocator.pt" ]]; then
  echo "NBS allocator state missing: $NBS_CHECKPOINT/nash_rank_allocator.pt"
  exit 2
fi
for path in "$NBS_TRAIN_LOG" "$UNIFORM_TRAIN_LOG"; do
  if [[ ! -f "$path" ]]; then
    echo "Source training log missing: $path"
    exit 2
  fi
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="viewport_prediction/data/experiment_runs/netllm_vs_nbs/selector_speculative_ablation/$RUN_ID"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > viewport_prediction/data/experiment_runs/netllm_vs_nbs/selector_speculative_ablation_latest.txt

COMMON_ARGS=(
  --test
  --train-dataset Jin2022
  --test-dataset Jin2022
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

run_case() {
  local number="$1"
  local case_name="$2"
  local variant="$3"
  local model_path="$4"
  local rank="$5"
  local inference_tag="$6"
  local train_log="$7"
  shift 7

  local output_dir="$RUN_DIR/${number}_${case_name}"
  mkdir -p "$output_dir/figures"
  echo "[$number/4] Starting $case_name"

  python run_plm.py \
    "${COMMON_ARGS[@]}" \
    --model-path "$model_path" \
    --rank "$rank" \
    --results-output-dir "$output_dir" \
    --latency-output-path "$output_dir/latency.json" \
    --inference-trace-output-path "$output_dir/inference_trace.json" \
    --inference-tag "$inference_tag" \
    "$@" \
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

  plot_args=(
    python analysis/plot_netllm_experiment.py
    --variant "$variant"
    --train-log "$train_log"
    --result-csv "$result_csv"
    --output-dir "$output_dir/figures"
    --latency-json "$output_dir/latency.json"
  )
  if [[ "$variant" == "nbs_v12_repeat" ]]; then
    plot_args+=(--allocator-state "$NBS_CHECKPOINT/nash_rank_allocator.pt")
  fi
  "${plot_args[@]}" 2>&1 | tee "$output_dir/plot.log"
}

NBS_ARGS=(
  --use-adalora
  --adalora-allocator nbs
  --adalora-rank-config configs/adalora_rank_config_llama7b_min4_max32.json
  --adalora-rank-budget 736
  --experiment-tag nbs_v12_repeat
)
UNIFORM_ARGS=(
  --lora-rank-config configs/lora_rank_pattern_llama7b_budget736.json
  --experiment-tag uniform_b736
)

run_case 1 nbs_selector nbs_v12_repeat "$NBS_CHECKPOINT" 32 selector "$NBS_TRAIN_LOG" \
  --selector-recent-k "${SELECTOR_RECENT_K:-6}" "${NBS_ARGS[@]}"

run_case 2 nbs_speculative nbs_v12_repeat "$NBS_CHECKPOINT" 32 speculative "$NBS_TRAIN_LOG" \
  --speculative-gamma "${SPECULATIVE_GAMMA:-4}" \
  --speculative-threshold "${SPECULATIVE_THRESHOLD:-0.3}" "${NBS_ARGS[@]}"

run_case 3 nbs_full_stack nbs_v12_repeat "$NBS_CHECKPOINT" 32 full_stack "$NBS_TRAIN_LOG" \
  --selector-recent-k "${SELECTOR_RECENT_K:-6}" \
  --speculative-gamma "${SPECULATIVE_GAMMA:-4}" \
  --speculative-threshold "${SPECULATIVE_THRESHOLD:-0.3}" "${NBS_ARGS[@]}"

run_case 4 uniform_b736_full_stack uniform_b736 "$UNIFORM_CHECKPOINT" 12 full_stack "$UNIFORM_TRAIN_LOG" \
  --selector-recent-k "${SELECTOR_RECENT_K:-6}" \
  --speculative-gamma "${SPECULATIVE_GAMMA:-4}" \
  --speculative-threshold "${SPECULATIVE_THRESHOLD:-0.3}" "${UNIFORM_ARGS[@]}"

python analysis/compare_selector_speculative_ablation.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/comparison"

printf '{"run_id":"%s","status":"complete","run_dir":"%s"}\n' \
  "$RUN_ID" "$RUN_DIR" > "$RUN_DIR/status.json"
echo "Selector/speculative ablation complete: $RUN_DIR"
