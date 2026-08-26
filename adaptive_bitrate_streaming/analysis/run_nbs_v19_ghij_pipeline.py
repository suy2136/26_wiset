"""Sequentially train and compact-test ABR NBS experiments G-H-I-J."""

try:
    from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
        RESULTS_ROOT, parse_args, run_group,
    )
except ModuleNotFoundError:
    from run_nbs_v19_group_pipeline import RESULTS_ROOT, parse_args, run_group


EXPERIMENTS = (
    {
        "name": "G", "rank_budget": 1536, "physical_rank": 32,
        "lr": 2.5e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "H", "rank_budget": 1536, "physical_rank": 32,
        "lr": 1.5e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "I", "rank_budget": 2048, "physical_rank": 64,
        "lr": 1e-4, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config_max64.json",
    },
    {
        "name": "J", "rank_budget": 2048, "physical_rank": 64,
        "lr": 7.5e-5, "warmup_steps": 500,
        "rank_config": "configs/nbs_v19_rank_config_max64.json",
    },
)


def main(argv=None):
    args = parse_args(
        argv,
        state_file=RESULTS_ROOT / "nbs_v19_ghij_state.json",
        output_file=RESULTS_ROOT / "nbs_v19_ghij_results.csv",
    )
    run_group(args, EXPERIMENTS)


if __name__ == "__main__":
    main()
