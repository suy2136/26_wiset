"""Train the follow-up ABR NBS experiments sequentially in E-F order."""

from pathlib import Path
import sys


ABR_ROOT = Path(__file__).resolve().parents[1]
if str(ABR_ROOT) not in sys.path:
    sys.path.insert(0, str(ABR_ROOT))

from analysis.run_nbs_v19_cad_experiments import (
    build_training_command,
    parse_args,
    run_experiments,
)


DEFAULT_STATE = (
    ABR_ROOT / "artifacts" / "results" / "nbs_v19_ef_training_state.json"
)

# E tests a middle-capacity/high-LR balance. F isolates D's LR stability.
EXPERIMENTS = (
    {
        "name": "E", "rank_budget": 2048, "physical_rank": 64,
        "lr": 2e-4, "lr_schedule": "cosine",
        "rank_config": "configs/nbs_v19_rank_config_max64.json",
    },
    {
        "name": "F", "rank_budget": 3072, "physical_rank": 64,
        "lr": 1e-4, "lr_schedule": "cosine",
        "rank_config": "configs/nbs_v19_rank_config_max64.json",
    },
)


def main(argv=None):
    args = parse_args(argv, default_state=DEFAULT_STATE)
    run_experiments(args, EXPERIMENTS)


if __name__ == "__main__":
    main()
