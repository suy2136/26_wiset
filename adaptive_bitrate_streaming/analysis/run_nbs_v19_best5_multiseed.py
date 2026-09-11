"""Run/aggregate the ABR best-five inference matrix over data seeds 1, 2, 3.

An existing seed-1 CSV is reused when available. Missing seeds are evaluated
with the same checkpoint and feature settings, and both per-seed records and
mean/sample-standard-deviation summaries are written after every seed.
"""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / "artifacts" / "results"
BEST5_SCRIPT = Path(__file__).with_name("run_nbs_v19_best5_inference.py")
DEFAULT_SEED1_RESULTS = RESULTS_ROOT / "nbs_v19_budget1536_best5_inference.csv"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "nbs_v19_budget1536_best5_multiseed"

SPEC = importlib.util.spec_from_file_location("abr_best5_inference", BEST5_SCRIPT)
best5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(best5)

AGGREGATE_METRICS = (
    "mean_reward", "qoe_raw_mean", "mean_bitrate_mbps",
    "mean_rebuffer_s_per_chunk", "total_rebuffer_s",
    "mean_smoothness_mbps", "inference_latency_mean_ms",
    "inference_latency_p50_ms", "inference_latency_p95_ms",
    "token_reduction_ratio", "temporal_history_reduction_ratio",
    "intra_token_reduction_ratio", "llm_call_reduction_ratio",
    "acceptance_rate", "target_plm_calls", "inference_calls",
)
CONFIG_FIELDS = (
    "configured_temporal", "configured_token", "configured_speculative",
    "configured_token_offsets", "configured_speculative_drafter",
    "rank_budget", "physical_rank",
)


def read_rows(path, data_seed):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {item["name"] for item in best5.EXPERIMENTS}
    names = [row.get("experiment") for row in rows]
    if len(rows) != len(expected) or set(names) != expected:
        raise ValueError(
            f"{path} must contain each best-five experiment exactly once"
        )
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate experiment rows in {path}")
    for row in rows:
        recorded = row.get("data_seed")
        if recorded not in (None, "", str(data_seed)):
            raise ValueError(
                f"{path} records data_seed={recorded}, expected {data_seed}"
            )
        row["data_seed"] = data_seed
        row["seed_results_path"] = str(path.resolve())
    return rows


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def aggregate_rows(rows, data_seeds):
    expected_seeds = set(data_seeds)
    grouped = {item["name"]: [] for item in best5.EXPERIMENTS}
    for row in rows:
        if row.get("experiment") in grouped:
            grouped[row["experiment"]].append(row)
    summaries = []
    for experiment in best5.EXPERIMENTS:
        name = experiment["name"]
        group = grouped[name]
        observed = {int(row["data_seed"]) for row in group}
        if observed != expected_seeds or len(group) != len(expected_seeds):
            raise ValueError(
                f"{name} has seeds {sorted(observed)}, expected {sorted(expected_seeds)}"
            )
        summary = {
            "experiment": name,
            "num_data_seeds": len(data_seeds),
            "data_seeds": ",".join(str(seed) for seed in data_seeds),
        }
        for field in CONFIG_FIELDS:
            if field in group[0]:
                summary[field] = group[0][field]
        for metric in AGGREGATE_METRICS:
            values = [finite_float(row.get(metric)) for row in group]
            if any(value is None for value in values):
                continue
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summaries.append(summary)

    baseline = next(
        row for row in summaries if row["experiment"] == "nbs_compact_only"
    )
    base_reward = baseline.get("mean_reward_mean")
    base_latency = baseline.get("inference_latency_mean_ms_mean")
    for row in summaries:
        reward = row.get("mean_reward_mean")
        latency = row.get("inference_latency_mean_ms_mean")
        if reward is not None and base_reward is not None:
            row["mean_reward_delta_vs_nbs"] = reward - base_reward
            row["mean_reward_change_ratio_vs_nbs"] = (
                0.0 if base_reward == 0 else reward / base_reward - 1.0
            )
        if latency and base_latency:
            row["inference_speedup_vs_nbs"] = base_latency / latency
            row["inference_latency_reduction_vs_nbs"] = (
                1.0 - latency / base_latency
            )
    return summaries


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir, all_rows, summaries, data_seeds):
    runs_path = output_dir / "best5_multiseed_runs.csv"
    summary_path = output_dir / "best5_multiseed_summary.csv"
    write_csv(runs_path, all_rows)
    write_csv(summary_path, summaries)
    (output_dir / "best5_multiseed_summary.json").write_text(
        json.dumps(
            {"data_seeds": data_seeds, "experiments": summaries},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return runs_path, summary_path


def seed_command(args, data_seed, output):
    command = [
        sys.executable, str(BEST5_SCRIPT),
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--base-model-dir", str(args.base_model_dir),
        "--exp-pool-path", str(args.exp_pool_path),
        "--rank-budget", str(args.rank_budget),
        "--physical-rank", str(args.physical_rank),
        "--rank-config", str(args.rank_config),
        "--trace", args.trace, "--trace-num", str(args.trace_num),
        "--video", args.video, "--device", args.device,
        "--data-seed", str(data_seed), "--output", str(output),
    ]
    if args.resume:
        command.append("--resume")
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, default=best5.DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path, default=best5.DEFAULT_EXP_POOL)
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
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seed1-results", type=Path, default=DEFAULT_SEED1_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_seeds = list(dict.fromkeys(args.data_seeds))
    if any(seed < 0 for seed in data_seeds):
        raise ValueError("data seeds must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for data_seed in data_seeds:
        generated = args.output_dir / f"best5_seed_{data_seed}.csv"
        source = (
            args.seed1_results
            if data_seed == 1 and args.seed1_results.is_file()
            else generated
        )
        if source.is_file():
            print(f"[seed {data_seed}] reusing {source.resolve()}", flush=True)
        else:
            command = seed_command(args, data_seed, generated)
            print(f"[seed {data_seed}] {' '.join(map(str, command))}", flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, cwd=ABR_ROOT, check=True)
            source = generated
        if not args.dry_run:
            all_rows.extend(read_rows(source, data_seed))
            completed = sorted({int(row["data_seed"]) for row in all_rows})
            if set(completed) == set(data_seeds):
                summaries = aggregate_rows(all_rows, data_seeds)
                runs_path, summary_path = write_outputs(
                    args.output_dir, all_rows, summaries, data_seeds
                )
                print(f"Per-seed results saved at: {runs_path.resolve()}")
                print(f"Mean/std summary saved at: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
