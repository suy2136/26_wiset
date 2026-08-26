"""Sequentially train and compact-test ABR NBS experiments K-L-M-N."""

try:
    from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
        RESULTS_ROOT, parse_args, run_group,
    )
except ModuleNotFoundError:
    from run_nbs_v19_group_pipeline import RESULTS_ROOT, parse_args, run_group


EXPERIMENTS = (
    {
        "name": "K", "rank_budget": 1792, "physical_rank": 32,
        "lr": 2e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "L", "rank_budget": 1792, "physical_rank": 32,
        "lr": 1.5e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "M", "rank_budget": 1824, "physical_rank": 32,
        "lr": 2e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "N", "rank_budget": 1824, "physical_rank": 32,
        "lr": 1.5e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
)


def main(argv=None):
    args = parse_args(
        argv,
        state_file=RESULTS_ROOT / "nbs_v19_klmn_state.json",
        output_file=RESULTS_ROOT / "nbs_v19_klmn_results.csv",
    )
    run_group(args, EXPERIMENTS)


if __name__ == "__main__":
    main()
