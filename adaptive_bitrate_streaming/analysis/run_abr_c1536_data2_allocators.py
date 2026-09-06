"""Run five capacity-matched ABR allocators with isolated data seed 2."""

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
    "data_seed": 2,
}

EXPERIMENTS = (
    {
        **COMMON,
        "name": "NBS_C1536_DATA2",
        "method": "nbs",
        "physical_rank": 32,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        **COMMON,
        "name": "UNIFORM_R24_DATA2",
        "method": "uniform_lora",
        "physical_rank": 24,
    },
    {
        **COMMON,
        "name": "ADALORA_C1536_DATA2",
        "method": "adalora",
        "physical_rank": 32,
        "allocation_interval": 10,
        "adalora_schedule_epochs": 20,
    },
    {
        **COMMON,
        "name": "EVA_C1536_DATA2",
        "method": "eva",
        "physical_rank": 32,
        "min_rank": 2,
        "max_rank": 32,
        "eva_metric": "ratio",
        "eva_similarity_threshold": 0.99,
        "eva_min_batches": 2,
        "eva_max_batches": 512,
        "eva_allow_unconverged": True,
    },
    {
        **COMMON,
        "name": "SHAPLEY_C1536_DATA2",
        "method": "shapley",
        "physical_rank": 32,
        "allocation_interval": 10,
        "adalora_schedule_epochs": 20,
        "shapley_permutations": 1,
        "shapley_validation_batches": 1,
        "shapley_truncate_fraction": 0.05,
        "shapley_antithetic": True,
    },
)


def main(argv=None):
    args = parse_args(
        argv,
        state_file=RESULTS_ROOT / "abr_c1536_data2_allocators_state.json",
        output_file=RESULTS_ROOT / "abr_c1536_data2_allocators_results.csv",
    )
    run_group(args, EXPERIMENTS)


if __name__ == "__main__":
    main()
