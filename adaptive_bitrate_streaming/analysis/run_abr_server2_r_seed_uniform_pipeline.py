"""Run R, a seed-2 C replication, and matched Uniform LoRA on server 2."""

try:
    from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
        RESULTS_ROOT, parse_args, run_group,
    )
except ModuleNotFoundError:
    from run_nbs_v19_group_pipeline import RESULTS_ROOT, parse_args, run_group


EXPERIMENTS = (
    {
        "name": "R", "rank_budget": 1664, "physical_rank": 32,
        "lr": 1.75e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "C_SEED2", "rank_budget": 1536, "physical_rank": 32,
        "lr": 2e-4, "warmup_steps": 500, "seed": 2,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "UNIFORM_R24", "method": "uniform_lora",
        "rank_budget": 1536, "physical_rank": 24,
        "lr": 2e-4, "warmup_steps": 500,
    },
)


def main(argv=None):
    args = parse_args(
        argv,
        state_file=RESULTS_ROOT / "abr_server2_r_seed_uniform_state.json",
        output_file=RESULTS_ROOT / "abr_server2_r_seed_uniform_results.csv",
    )
    run_group(args, EXPERIMENTS)


if __name__ == "__main__":
    main()
