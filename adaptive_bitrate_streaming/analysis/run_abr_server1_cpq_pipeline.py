"""Run the three ABR NBS experiments assigned to server 1."""

try:
    from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
        RESULTS_ROOT, parse_args, run_group,
    )
except ModuleNotFoundError:
    from run_nbs_v19_group_pipeline import RESULTS_ROOT, parse_args, run_group


EXPERIMENTS = (
    {
        "name": "C_REPRO", "rank_budget": 1536, "physical_rank": 32,
        "lr": 2e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "P", "rank_budget": 1536, "physical_rank": 32,
        "lr": 1.75e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "Q", "rank_budget": 1664, "physical_rank": 32,
        "lr": 2e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
)


def main(argv=None):
    args = parse_args(
        argv,
        state_file=RESULTS_ROOT / "abr_server1_cpq_state.json",
        output_file=RESULTS_ROOT / "abr_server1_cpq_results.csv",
    )
    run_group(args, EXPERIMENTS)


if __name__ == "__main__":
    main()
