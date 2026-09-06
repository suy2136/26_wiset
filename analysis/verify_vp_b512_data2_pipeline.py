"""Static smoke verification for the VP budget-512/data-seed2 pipeline."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "scripts" / "run_netllm_experiment.sh").read_text(
    encoding="utf-8"
)
SEQUENCE = (ROOT / "scripts" / "run_vp_b512_data2_allocators.sh").read_text(
    encoding="utf-8"
)
RECOVERY = (
    ROOT / "scripts" / "resume_vp_b512_data2_after_stock.sh"
).read_text(encoding="utf-8")
LOW_RANK = (ROOT / "models" / "low_rank.py").read_text(encoding="utf-8")


def require(text: str, source: str) -> None:
    if text not in source:
        raise AssertionError(f"missing pipeline contract: {text}")


for variant in (
    "uniform_r8_data2",
    "adalora_b512_data2",
    "eva_b512_data2",
    "shapley_b512_data2",
):
    require(variant, RUNNER)
    require(f"run_netllm_experiment.sh {variant}", SEQUENCE)

for setting in (
    "export EPOCHS=4",
    "export GRAD_ACCUM_STEPS=32",
    "export LEARNING_RATE=0.0002",
    "export SEED=1",
    "export LORA_SEED=1",
    "export DATA_SEED=2",
):
    require(setting, SEQUENCE)

require('RANK=8\n  RANK_BUDGET=512', RUNNER)
require('--adalora-init-rank "$ADALORA_INIT_RANK"', RUNNER)
require('ADALORA_INIT_RANK=32', RUNNER)
require('MODEL_TAG="llama_base_low_rank_adalora_adalora_b512_data2"', RUNNER)
require('MODEL_TAG="llama_base_low_rank_eva_b512_data2"', RUNNER)
require('MODEL_TAG="llama_base_low_rank_adalora_shapley_b512_data2"', RUNNER)
require('EVA_MIN_RANK=2', RUNNER)
require('EVA_MAX_RANK=32', RUNNER)
require('physical_rank < int(rank)', LOW_RANK)
require('find viewport_prediction/data/ft_plms -type d', RECOVERY)
require('-name best_ar_model', RECOVERY)
require('--adalora-init-rank 32', RECOVERY)
require('--results-output-dir "$generated_dir"', RECOVERY)
require('run_netllm_experiment.sh eva_b512_data2', RECOVERY)
require('run_netllm_experiment.sh shapley_b512_data2', RECOVERY)
require('export SKIP_VISUALIZATION="${SKIP_VISUALIZATION:-1}"', RECOVERY)
require('SKIP_VISUALIZATION="${SKIP_VISUALIZATION:-0}"', RUNNER)
require('server-side visualization skipped', RUNNER)
require('resume-after-stock', SEQUENCE)

positions = [
    SEQUENCE.index(f"run_netllm_experiment.sh {variant}")
    for variant in (
        "uniform_r8_data2",
        "adalora_b512_data2",
        "eva_b512_data2",
        "shapley_b512_data2",
    )
]
if positions != sorted(positions):
    raise AssertionError("allocator sequence order changed")

print("[PASS] VP budget512/data-seed2 four-allocator pipeline wiring")
