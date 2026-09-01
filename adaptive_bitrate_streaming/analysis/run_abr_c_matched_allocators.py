"""Run C-capacity-matched AdaLoRA, EVA, and Shapley ABR experiments."""

from __future__ import annotations

import subprocess
import sys

try:
    from adaptive_bitrate_streaming.analysis.run_nbs_v19_group_pipeline import (
        RESULTS_ROOT,
        parse_args,
        run_group,
    )
except ModuleNotFoundError:
    from run_nbs_v19_group_pipeline import RESULTS_ROOT, parse_args, run_group


EXPERIMENTS = (
    {
        "name": "ADALORA_C1536",
        "method": "adalora",
        "rank_budget": 1536,
        "physical_rank": 32,
        "lr": 2e-4,
        "warmup_steps": 500,
        "allocation_interval": 10,
        "adalora_schedule_epochs": 20,
    },
    {
        "name": "EVA_C1536",
        "method": "eva",
        "rank_budget": 1536,
        "physical_rank": 32,
        "min_rank": 2,
        "max_rank": 32,
        "lr": 2e-4,
        "warmup_steps": 500,
        "eva_metric": "ratio",
        "eva_similarity_threshold": 0.99,
        "eva_min_batches": 2,
        "eva_max_batches": 128,
    },
    {
        "name": "SHAPLEY_C1536",
        "method": "shapley",
        "rank_budget": 1536,
        "physical_rank": 32,
        "lr": 2e-4,
        "warmup_steps": 500,
        "allocation_interval": 10,
        "adalora_schedule_epochs": 20,
        # Deliberately retain the previously validated low-cost estimator.
        "shapley_permutations": 1,
        "shapley_validation_batches": 1,
        "shapley_truncate_fraction": 0.05,
        "shapley_antithetic": True,
    },
)


def run_experiments(argv, experiments, result_stem):
    output = RESULTS_ROOT / f"{result_stem}_results.csv"
    args = parse_args(
        argv,
        state_file=RESULTS_ROOT / f"{result_stem}_state.json",
        output_file=output,
    )
    run_group(args, experiments)
    if not args.dry_run and output.is_file():
        subprocess.run(
            [
                sys.executable,
                "analysis/plot_nbs_v19_inference_results.py",
                str(output.resolve()),
                "--output-dir",
                str((RESULTS_ROOT / f"{result_stem}_plots").resolve()),
                "--prefix",
                result_stem,
            ],
            check=True,
        )


def main(argv=None):
    run_experiments(argv, EXPERIMENTS, "abr_c_matched_allocators")


if __name__ == "__main__":
    main()
