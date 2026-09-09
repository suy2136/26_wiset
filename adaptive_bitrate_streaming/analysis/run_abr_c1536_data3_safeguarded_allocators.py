"""Run C-matched ABR allocators with LoRA seed 1 and data seed 3."""

from __future__ import annotations

try:
    from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
        RESULTS_ROOT,
        parse_args,
        run_group,
    )
except ModuleNotFoundError:
    from run_nbs_v19_group_pipeline import RESULTS_ROOT, parse_args, run_group


COMMON = {
    "rank_budget": 1536,
    "lr": 2e-4,
    "warmup_steps": 500,
    "seed": 1,
    "lora_seed": 1,
    "data_seed": 3,
    "run_tag": "fp16safe_data3_v1",
}

EXPERIMENTS = (
    {
        **COMMON, "name": "NBS_C1536_DATA3_SAFE", "method": "nbs",
        "physical_rank": 32,
        "rank_config": "configs/nbs_v19_rank_config.json",
        "nbs_allocation_audit": True,
    },
    {
        **COMMON, "name": "UNIFORM_R24_DATA3_SAFE",
        "method": "uniform_lora", "physical_rank": 24,
    },
    {
        **COMMON, "name": "ADALORA_C1536_DATA3_SAFE", "method": "adalora",
        "physical_rank": 32, "allocation_interval": 10,
        "adalora_schedule_epochs": 20,
    },
    {
        **COMMON, "name": "EVA_C1536_DATA3_SAFE", "method": "eva",
        "physical_rank": 32, "min_rank": 2, "max_rank": 32,
        "eva_metric": "ratio", "eva_similarity_threshold": 0.99,
        "eva_min_batches": 2, "eva_max_batches": 512,
        "eva_allow_unconverged": True,
    },
    {
        **COMMON, "name": "SHAPLEY_C1536_DATA3_SAFE", "method": "shapley",
        "physical_rank": 32, "allocation_interval": 10,
        "adalora_schedule_epochs": 20, "shapley_permutations": 1,
        "shapley_validation_batches": 1,
        "shapley_truncate_fraction": 0.05,
        "shapley_antithetic": True,
    },
)


def run_experiments(argv, experiments, output_stem):
    args = parse_args(
        argv,
        state_file=RESULTS_ROOT / f"{output_stem}_state.json",
        output_file=RESULTS_ROOT / f"{output_stem}_results.csv",
    )
    run_group(args, experiments)


def main(argv=None):
    run_experiments(
        argv, EXPERIMENTS, "abr_c1536_data3_safeguarded_allocators"
    )


if __name__ == "__main__":
    main()
