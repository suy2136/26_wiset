"""Confirm the best isolated ABR inference modules over seeds 1--3.

The seed-1 isolated sweep and the already completed three-seed baseline/full
combination are reused.  Only temporal K=1, token offsets 0/2/3/4/7, and
repeat-last speculative K=5 are evaluated for seeds 2 and 3.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics


ABR_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = Path(__file__).with_name("run_nbs_v19_module_sweep_pipeline.py")
SPEC = importlib.util.spec_from_file_location("abr_module_sweep", SWEEP_SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)

DEFAULT_SOURCE_DIR = sweep.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = (
    sweep.best5.RESULTS_ROOT / "nbs_v19_budget1536_best_modules_multiseed"
)
COMBINED_NAME = (
    "combined__temporal_k1__token_o0_2_3_4_7__spec_repeat_last_k5"
)
TARGET_SPECS = (
    sweep.baseline_spec(),
    {
        "name": "temporal_k1", "module": "temporal",
        "temporal": True, "event_max_events": 1,
    },
    {
        "name": "token_o0_2_3_4_7", "module": "token",
        "token": True, "token_offsets": (0, 2, 3, 4, 7),
    },
    {
        "name": "spec_repeat_last_k5", "module": "speculative",
        "speculative": True, "speculative_steps": 5,
        "speculative_drafter": "repeat-last",
    },
    {
        "name": COMBINED_NAME, "module": "combined",
        "temporal": True, "event_max_events": 1,
        "token": True, "token_offsets": (0, 2, 3, 4, 7),
        "speculative": True, "speculative_steps": 5,
        "speculative_drafter": "repeat-last",
    },
)
RUN_SPECS = TARGET_SPECS[1:4]


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalized(value):
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def validate_row(row, spec, seed, rank_budget, physical_rank):
    if row.get("experiment") != spec["name"]:
        raise ValueError(f"unexpected experiment row: {row.get('experiment')}")
    if int(float(row.get("data_seed", -1))) != seed:
        raise ValueError(f"{spec['name']} has the wrong data seed")
    if int(float(row.get("rank_budget", -1))) != rank_budget:
        raise ValueError(f"{spec['name']} has the wrong rank budget")
    if int(float(row.get("physical_rank", -1))) != physical_rank:
        raise ValueError(f"{spec['name']} has the wrong physical rank")
    expected = sweep.configured_fields(spec)
    for field, value in expected.items():
        if normalized(row.get(field)) != normalized(value):
            raise ValueError(
                f"{spec['name']} configuration mismatch for {field}: "
                f"expected {value!r}, found {row.get(field)!r}"
            )


def reusable_rows(args):
    independent = read_csv(args.independent_seed1)
    confirmation = read_csv(args.confirmation_runs)
    rows = []
    # The three isolated settings reuse only their seed-1 screening records.
    for spec in RUN_SPECS:
        matches = [
            row for row in independent
            if row.get("experiment") == spec["name"]
        ]
        if len(matches) != 1:
            raise ValueError(f"seed-1 source must contain exactly one {spec['name']}")
        validate_row(matches[0], spec, 1, args.rank_budget, args.physical_rank)
        rows.append(matches[0])

    # Baseline and the requested full combination have already been confirmed.
    for spec in (TARGET_SPECS[0], TARGET_SPECS[4]):
        for seed in (1, 2, 3):
            matches = [
                row for row in confirmation
                if row.get("experiment") == spec["name"]
                and int(float(row.get("data_seed", -1))) == seed
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"confirmation source must contain one {spec['name']} seed {seed}"
                )
            validate_row(
                matches[0], spec, seed, args.rank_budget, args.physical_rank
            )
            rows.append(matches[0])
    return rows


def aggregate(rows):
    summaries = []
    for spec in TARGET_SPECS:
        group = sorted(
            (row for row in rows if row["experiment"] == spec["name"]),
            key=lambda row: int(row["data_seed"]),
        )
        seeds = [int(row["data_seed"]) for row in group]
        if seeds != [1, 2, 3]:
            raise ValueError(f"{spec['name']} has seeds {seeds}, expected [1, 2, 3]")
        summary = {
            "experiment": spec["name"], "num_seeds": 3,
            "data_seeds": "1,2,3", **sweep.configured_fields(spec),
        }
        for metric in sweep.SUMMARY_METRICS:
            values = [sweep.finite_number(row.get(metric)) for row in group]
            if all(value is not None for value in values):
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(summary)
    baseline = summaries[0]
    for row in summaries:
        row["mean_reward_change_ratio_vs_nbs"] = (
            row["mean_reward_mean"] / baseline["mean_reward_mean"] - 1.0
        )
        row["inference_latency_reduction_vs_nbs"] = (
            1.0 - row["inference_latency_mean_ms_mean"]
            / baseline["inference_latency_mean_ms_mean"]
        )
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-dir", type=Path, default=sweep.best5.DEFAULT_BASE_MODEL
    )
    parser.add_argument(
        "--exp-pool-path", type=Path, default=sweep.best5.DEFAULT_EXP_POOL
    )
    parser.add_argument("--rank-budget", type=int, default=1536)
    parser.add_argument("--physical-rank", type=int, default=32)
    parser.add_argument(
        "--rank-config", type=Path,
        default=Path("configs/nbs_v19_rank_config.json"),
    )
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--independent-seed1", type=Path,
        default=DEFAULT_SOURCE_DIR / "independent_seed1.csv",
    )
    parser.add_argument(
        "--confirmation-runs", type=Path,
        default=DEFAULT_SOURCE_DIR / "confirmation_runs.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        sweep.best5.validate_checkpoint(args.checkpoint_dir, args.rank_budget)
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")

    runs_path = args.output_dir / "best_modules_multiseed_runs.csv"
    if args.dry_run:
        rows = []
    else:
        rows = reusable_rows(args)
        previous = sweep.load_rows(runs_path) if args.resume else []
        rows.extend(row for row in previous if int(row["data_seed"]) in (2, 3))
        sweep.write_rows(runs_path, rows)

    for seed in (2, 3):
        rows = sweep.run_specs(args, RUN_SPECS, seed, runs_path, rows)
    if args.dry_run:
        return

    summaries = aggregate(rows)
    summary_path = args.output_dir / "best_modules_multiseed_summary.csv"
    sweep.write_rows(summary_path, summaries)
    print(f"Per-seed results: {runs_path.resolve()}", flush=True)
    print(f"Three-seed summary: {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
