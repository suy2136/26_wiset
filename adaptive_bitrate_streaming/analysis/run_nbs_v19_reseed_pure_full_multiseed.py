"""Evaluate pure NBS and the established full stack with episode reseeding."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


ABR_ROOT = Path(__file__).resolve().parents[1]
BEST5_SCRIPT = Path(__file__).with_name("run_nbs_v19_best5_inference.py")
DEFAULT_OUTPUT_DIR = (
    ABR_ROOT / "artifacts" / "results"
    / "nbs_v19_budget1536_reseed_pure_full_multiseed"
)
EXPERIMENTS = ("nbs_compact_only", "combined_temporal_token_spec")
METRICS = (
    "mean_reward", "qoe_raw_mean", "mean_bitrate_mbps",
    "mean_rebuffer_s_per_chunk", "total_rebuffer_s",
    "mean_smoothness_mbps", "inference_latency_mean_ms",
    "inference_latency_p50_ms", "inference_latency_p95_ms",
    "token_reduction_ratio", "temporal_history_reduction_ratio",
    "intra_token_reduction_ratio", "llm_call_reduction_ratio",
    "acceptance_rate", "target_plm_calls", "inference_calls",
)


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_seed_rows(path, seed):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    names = [row.get("experiment") for row in rows]
    if len(rows) != 2 or set(names) != set(EXPERIMENTS):
        raise ValueError(f"{path} must contain exactly {EXPERIMENTS}")
    for row in rows:
        if row.get("evaluation_rng_mode") != "per-episode":
            raise ValueError(f"{path} is not a per-episode-reseed result")
        row["data_seed"] = seed
        row["seed_results_path"] = str(path.resolve())
    return rows


def seed_command(args, seed, output):
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
        "--data-seed", str(seed), "--output", str(output),
        "--evaluation-rng-mode", "per-episode",
        "--only", *EXPERIMENTS,
    ]
    if args.resume:
        command.append("--resume")
    return command


def aggregate(rows, seeds):
    summaries = []
    for name in EXPERIMENTS:
        group = sorted(
            (row for row in rows if row["experiment"] == name),
            key=lambda row: int(row["data_seed"]),
        )
        if [int(row["data_seed"]) for row in group] != sorted(seeds):
            raise ValueError(f"incomplete seed set for {name}")
        summary = {
            "experiment": name,
            "evaluation_rng_mode": "per-episode",
            "num_data_seeds": len(seeds),
            "data_seeds": ",".join(map(str, sorted(seeds))),
        }
        for metric in METRICS:
            values = [finite_float(row.get(metric)) for row in group]
            if any(value is None for value in values):
                continue
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summaries.append(summary)

    baseline = summaries[0]
    for summary in summaries:
        reward = summary["mean_reward_mean"]
        latency = summary["inference_latency_mean_ms_mean"]
        summary["mean_reward_delta_vs_nbs"] = (
            reward - baseline["mean_reward_mean"]
        )
        summary["mean_reward_change_ratio_vs_nbs"] = (
            reward / baseline["mean_reward_mean"] - 1.0
        )
        summary["inference_latency_reduction_vs_nbs"] = (
            1.0 - latency / baseline["inference_latency_mean_ms_mean"]
        )

    baseline_by_seed = {
        int(row["data_seed"]): row for row in rows
        if row["experiment"] == "nbs_compact_only"
    }
    full = summaries[1]
    paired_reward = []
    paired_rebuffer = []
    for row in rows:
        if row["experiment"] != "combined_temporal_token_spec":
            continue
        base = baseline_by_seed[int(row["data_seed"])]
        paired_reward.append(
            finite_float(row["mean_reward"]) - finite_float(base["mean_reward"])
        )
        paired_rebuffer.append(
            finite_float(row["total_rebuffer_s"])
            - finite_float(base["total_rebuffer_s"])
        )
    full["paired_qoe_delta_mean"] = statistics.mean(paired_reward)
    full["paired_qoe_delta_std"] = statistics.stdev(paired_reward)
    full["paired_total_rebuffer_delta_s_mean"] = statistics.mean(paired_rebuffer)
    full["paired_total_rebuffer_delta_s_std"] = statistics.stdev(paired_rebuffer)
    return summaries


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--exp-pool-path", type=Path, required=True)
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
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seeds = list(dict.fromkeys(args.evaluation_seeds))
    if len(seeds) < 2:
        raise ValueError("at least two evaluation seeds are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in seeds:
        output = args.output_dir / f"reseed_seed_{seed}.csv"
        seed_rows = None
        if args.resume and output.is_file():
            try:
                seed_rows = read_seed_rows(output, seed)
                print(
                    f"[seed {seed}] reusing verified reseed result {output}",
                    flush=True,
                )
            except ValueError as exc:
                print(
                    f"[seed {seed}] incomplete/incompatible result; resuming: {exc}",
                    flush=True,
                )
        if seed_rows is None:
            command = seed_command(args, seed, output)
            print(f"[seed {seed}] {' '.join(map(str, command))}", flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, cwd=ABR_ROOT, check=True)
            seed_rows = read_seed_rows(output, seed)
        if not args.dry_run:
            all_rows.extend(seed_rows)

    if args.dry_run:
        return
    summaries = aggregate(all_rows, seeds)
    runs_path = args.output_dir / "reseed_pure_full_per_seed.csv"
    summary_path = args.output_dir / "reseed_pure_full_summary.csv"
    write_csv(runs_path, all_rows)
    write_csv(summary_path, summaries)
    (args.output_dir / "reseed_pure_full_summary.json").write_text(
        json.dumps({"seeds": seeds, "results": summaries}, indent=2),
        encoding="utf-8",
    )
    print(f"Per-seed results: {runs_path.resolve()}")
    print(f"Three-seed summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
