"""Evaluate only C, A, and D ABR NBS checkpoints under one test setup."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / "artifacts" / "results"
DEFAULT_BASE_MODEL = ABR_ROOT.parent / "downloaded_plms" / "llama" / "base"
DEFAULT_EXP_POOL = ABR_ROOT / "artifacts" / "exp_pools" / "exp_pool.pkl"
DEFAULT_TRAINING_STATE = RESULTS_ROOT / "nbs_v19_cad_training_state.json"
DEFAULT_OUTPUT = RESULTS_ROOT / "nbs_v19_cad_test.csv"

EXPERIMENTS = (
    {
        "name": "C", "rank_budget": 1536, "mean_active_rank": 24.0,
        "physical_rank": 32,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "A", "rank_budget": 2048, "mean_active_rank": 32.0,
        "physical_rank": 32,
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "D", "rank_budget": 3072, "mean_active_rank": 48.0,
        "physical_rank": 64,
        "rank_config": "configs/nbs_v19_rank_config_max64.json",
    },
)


def checkpoints_from_state(path):
    if not path.is_file():
        raise FileNotFoundError(f"C/A/D training state not found: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    expected_order = [item["name"] for item in EXPERIMENTS]
    if state.get("order") != expected_order:
        raise ValueError("training state does not use C-A-D order")
    checkpoints = {}
    for experiment in EXPERIMENTS:
        name = experiment["name"]
        run = state.get("runs", {}).get(name)
        if not run or not run.get("checkpoint_dir"):
            raise ValueError(f"training state has no completed {name} run")
        checkpoints[name] = Path(run["checkpoint_dir"])
    return checkpoints


def validate_checkpoint(path, experiment):
    required = (
        "adapter_config.json", "modules_except_plm.bin",
        "nash_rank_allocator.pt", "checkpoint_metadata.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if not any((path / name).is_file() for name in (
        "adapter_model.bin", "adapter_model.safetensors"
    )):
        missing.append("adapter_model.bin or adapter_model.safetensors")
    if missing:
        raise FileNotFoundError(
            f"incomplete {experiment['name']} checkpoint: {', '.join(missing)}"
        )
    metadata = json.loads(
        (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("variant") != "nbs_v19" or metadata.get("seed") != 1:
        raise ValueError(f"{experiment['name']} is not a seed-1 NBS v19 checkpoint")
    if metadata.get("effective_rank_budget") != experiment["rank_budget"]:
        raise ValueError(
            f"{experiment['name']} budget mismatch: "
            f"{metadata.get('effective_rank_budget')} != {experiment['rank_budget']}"
        )
    adapter = json.loads(
        (path / "adapter_config.json").read_text(encoding="utf-8")
    )
    physical_rank = int(adapter.get("init_r", adapter.get("r", -1)))
    if physical_rank != experiment["physical_rank"]:
        raise ValueError(
            f"{experiment['name']} physical rank mismatch: "
            f"{physical_rank} != {experiment['physical_rank']}"
        )
    return metadata


def build_test_command(args, experiment, checkpoint):
    return [
        sys.executable, "run_plm.py", "--test", "--nbs-v19", "--fp16",
        "--seed", "1", "--plm-type", "llama", "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--model-dir", str(checkpoint.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--rank", str(experiment["physical_rank"]),
        "--nbs-rank-budget", str(experiment["rank_budget"]),
        "--nbs-rank-config", experiment["rank_config"],
        "--trace", args.trace, "--trace-num", str(args.trace_num),
        "--video", args.video, "--fixed-order",
        "--device", args.device, "--device-out", args.device,
        "--temporal-selector", "none", "--token-selector", "none",
        "--speculative-draft-steps", "0",
    ]


def newest_metrics(started_at):
    candidates = [
        path for path in RESULTS_ROOT.rglob("selector_metrics.json")
        if path.stat().st_mtime >= started_at
    ]
    if not candidates:
        raise RuntimeError("test produced no selector_metrics.json")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalar_metrics(metrics):
    return {
        key: value for key, value in metrics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def add_comparisons(rows):
    baseline = next((row for row in rows if row["experiment"] == "C"), None)
    if baseline is None:
        return rows
    baseline_qoe = float(baseline["mean_reward"])
    baseline_latency = float(baseline["inference_latency_mean_ms"])
    for row in rows:
        row["mean_reward_delta_vs_c"] = float(row["mean_reward"]) - baseline_qoe
        row["mean_reward_percent_vs_c"] = (
            float(row["mean_reward"]) / baseline_qoe - 1.0
        )
        row["inference_latency_reduction_vs_c"] = (
            1.0 - float(row["inference_latency_mean_ms"]) / baseline_latency
        )
    return rows


def run_signature(args, checkpoints):
    return {
        "experiments": list(EXPERIMENTS),
        "checkpoints": {name: str(path.resolve()) for name, path in checkpoints.items()},
        "base_model_dir": str(args.base_model_dir.resolve()),
        "exp_pool_path": str(args.exp_pool_path.resolve()),
        "trace": args.trace, "trace_num": args.trace_num,
        "video": args.video, "device": args.device, "seed": 1,
        "features": {"temporal": "none", "token": "none", "speculative": 0},
    }


def write_results(rows, output, signature):
    rows = add_comparisons(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "experiment", "rank_budget", "mean_active_rank", "physical_rank",
        "mean_reward", "mean_reward_delta_vs_c", "mean_reward_percent_vs_c",
        "inference_latency_mean_ms", "inference_latency_reduction_vs_c",
        "token_reduction_ratio", "checkpoint_dir", "metrics_path", "seed",
    ]
    fields = [key for key in preferred if key in fields] + [
        key for key in fields if key not in preferred
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_resume(output, signature):
    json_path = output.with_suffix(".json")
    manifest_path = output.with_suffix(".manifest.json")
    if not json_path.is_file() or not manifest_path.is_file():
        return []
    if json.loads(manifest_path.read_text(encoding="utf-8")) != signature:
        raise ValueError("saved C/A/D test state does not match current conditions")
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    for row in rows:
        for key in ("mean_reward", "inference_latency_mean_ms"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"saved {row['experiment']} {key} is non-finite")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-state", type=Path, default=DEFAULT_TRAINING_STATE)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path, default=DEFAULT_EXP_POOL)
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoints = checkpoints_from_state(args.training_state)
    signature = run_signature(args, checkpoints)
    metadata = {}
    if not args.dry_run:
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
        for experiment in EXPERIMENTS:
            metadata[experiment["name"]] = validate_checkpoint(
                checkpoints[experiment["name"]], experiment
            )
    rows = load_resume(args.output, signature) if args.resume else []
    completed = {row["experiment"] for row in rows}
    for experiment in EXPERIMENTS:
        name = experiment["name"]
        if name in completed:
            print(f"[{name}] already complete; skipping", flush=True)
            continue
        checkpoint = checkpoints[name]
        command = build_test_command(args, experiment, checkpoint)
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        started_at = time.time() - 1.0
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = newest_metrics(started_at)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "experiment": name, "seed": 1,
            "rank_budget": experiment["rank_budget"],
            "mean_active_rank": experiment["mean_active_rank"],
            "physical_rank": experiment["physical_rank"],
            "checkpoint_role": metadata[name].get("role"),
            "checkpoint_dir": str(checkpoint.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            **scalar_metrics(metrics),
        })
        write_results(rows, args.output, signature)
    if not args.dry_run:
        print(f"Results saved at: {args.output.resolve()}")


if __name__ == "__main__":
    main()
