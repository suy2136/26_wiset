"""Static verification for the VP budget-512/data-seed1 pipeline."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "scripts" / "run_netllm_experiment.sh").read_text(encoding="utf-8")
SEQUENCE = (ROOT / "scripts" / "run_vp_b512_data1_allocators.sh").read_text(
    encoding="utf-8"
)
CLI = (ROOT / "run_plm.py").read_text(encoding="utf-8")


def require(text: str, source: str) -> None:
    if text not in source:
        raise AssertionError(f"missing pipeline contract: {text}")


variants = ("uniform_r8_data1", "adalora_b512_data1", "eva_b512_data1")
for variant in variants:
    require(variant, RUNNER)
    require(variant, CLI)
    require(f"run_netllm_experiment.sh {variant}", SEQUENCE)

for setting in (
    "export EPOCHS=4",
    "export GRAD_ACCUM_STEPS=32",
    "export LEARNING_RATE=0.0002",
    "export SEED=1",
    "export LORA_SEED=1",
    "export DATA_SEED=1",
    'export SKIP_VISUALIZATION="${SKIP_VISUALIZATION:-1}"',
    'export EVA_MAX_BATCHES="${EVA_MAX_BATCHES:-512}"',
    'export EVA_ALLOW_UNCONVERGED="${EVA_ALLOW_UNCONVERGED:-1}"',
):
    require(setting, SEQUENCE)

require('RANK=8\n  RANK_BUDGET=512', RUNNER)
require('ADALORA_INIT_RANK=32', RUNNER)
require('EVA_MIN_RANK=2', RUNNER)
require('EVA_MAX_RANK=32', RUNNER)

positions = [SEQUENCE.index(f"run_netllm_experiment.sh {variant}") for variant in variants]
if positions != sorted(positions):
    raise AssertionError("allocator sequence order changed")

print("[PASS] VP budget512/data-seed1 three-allocator pipeline wiring")
