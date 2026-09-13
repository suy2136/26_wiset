"""Evaluate four non-NBS ABR LoRA methods under one safe protocol.

The supplied checkpoints are opened read-only.  No training, adapter saving,
rank allocation, or NBS compaction is performed.  Uniform LoRA is already a
physical rank-24 adapter (64 * 24 = 1536); AdaLoRA and Shapley are physically
resized from their saved rank patterns; EVA uses its saved heterogeneous
rank pattern.  All four therefore have an effective rank budget of 1536.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ABR_ROOT / "artifacts" / "results"
DEFAULT_OUTPUT_DIR = (
    RESULTS_ROOT / "abr_lora_methods_budget1536_fp16_prescaled_qk_multiseed"
)
SEEDS = (1, 2, 3)
METHODS = {
    "uniform": {
        "label": "Uniform LoRA r24", "variant": "uniform_lora",
        "role": "best", "physical_rank": 24,
    },
    "adalora": {
        "label": "Stock AdaLoRA", "variant": "adalora",
        "role": "final", "physical_rank": 32,
    },
    "shapley": {
        "label": "ShapLoRA", "variant": "shapley",
        "role": "final", "physical_rank": 32,
    },
    "eva": {
        "label": "EVA", "variant": "eva",
        "role": "best", "physical_rank": 32,
    },
}
SUMMARY_METRICS = (
    "mean_reward", "qoe_raw_mean", "mean_bitrate_mbps",
    "mean_rebuffer_s_per_chunk", "total_rebuffer_s",
    "mean_smoothness_mbps", "inference_latency_mean_ms",
    "inference_latency_p50_ms", "inference_latency_p95_ms",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform-checkpoint", type=Path, required=True)
    parser.add_argument("--adalora-checkpoint", type=Path, required=True)
    parser.add_argument("--shapley-checkpoint", type=Path, required=True)
    parser.add_argument("--eva-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--exp-pool-path", type=Path, required=True)
    parser.add_argument("--rank-budget", type=int, default=1536)
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def checkpoint_paths(args):
    return {
        method: getattr(args, f"{method}_checkpoint").resolve()
        for method in METHODS
    }


def validate_checkpoint(path, method, rank_budget):
    spec = METHODS[method]
    required = [
        "adapter_config.json", "modules_except_plm.bin",
        "checkpoint_metadata.json",
    ]
    if method == "eva":
        required.append("eva_state.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if not any((path / name).is_file() for name in (
        "adapter_model.bin", "adapter_model.safetensors",
    )):
        missing.append("adapter_model.bin or adapter_model.safetensors")
    if missing:
        raise FileNotFoundError(
            f"incomplete {method} checkpoint: {', '.join(missing)}"
        )
    metadata = json.loads(
        (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    expected = {
        "variant": spec["variant"], "role": spec["role"],
        "effective_rank_budget": rank_budget,
        "physical_rank": spec["physical_rank"], "seed": 1,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items() if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{method} checkpoint metadata mismatch: {mismatches}")
    return metadata


def build_command(args, method, checkpoint, seed):
    spec = METHODS[method]
    command = [
        sys.executable, "run_plm.py", "--test", "--fp16",
        "--seed", str(seed), "--lora-seed", "1", "--data-seed", str(seed),
        "--plm-type", "llama", "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--model-dir", str(checkpoint),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--rank", str(spec["physical_rank"]),
        "--trace", args.trace, "--trace-num", str(args.trace_num),
        "--video", args.video, "--fixed-order",
        "--device", args.device, "--device-out", args.device,
        "--evaluation-rng-mode", "per-episode",
        "--run-tag", "lora_methods_fp16_prescaled_qk",
        "--temporal-selector", "none", "--token-selector", "none",
        "--speculative-draft-steps", "0",
        "--fp16-numeric-safeguards", "--fp16-selective-clamp",
        "--fp16-selective-clamp-threshold", "60000.0",
        "--fp16-attention-prescaled-qk",
    ]
    if method == "uniform":
        command.extend(["--lora-method", "uniform"])
    elif method in ("adalora", "shapley"):
        command.extend([
            "--lora-method", method,
            "--adalora-rank-budget", str(args.rank_budget),
            "--adalora-allocation-interval", "10",
        ])
        if method == "shapley":
            command.extend([
                "--shapley-permutations", "1",
                "--shapley-validation-batches", "1",
                "--shapley-truncate-fraction", "0.05",
            ])
    elif method == "eva":
        command.extend([
            "--lora-method", "eva",
            "--eva-state-path", str(checkpoint / "eva_state.pt"),
        ])
    return command


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def scalar_metrics(metrics):
    return {
        key: value for key, value in metrics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def newest_metrics(started_at):
    candidates = [
        path for path in RESULTS_ROOT.rglob("selector_metrics.json")
        if path.stat().st_mtime >= started_at
    ]
    if not candidates:
        raise RuntimeError("evaluation produced no selector_metrics.json")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = [
        "method", "label", "evaluation_seed", "rank_budget",
        "physical_rank", "checkpoint_dir", "mean_reward", "qoe_raw_mean",
        "inference_latency_mean_ms", "inference_latency_p50_ms",
        "inference_latency_p95_ms", "total_rebuffer_s",
        "mean_rebuffer_s_per_chunk", "mean_bitrate_mbps",
        "mean_smoothness_mbps", "attention_score_mode",
        "fp16_prescaled_qk_layer_calls",
        "fp16_prescaled_qk_fallback_plm_calls",
        "fp32_attention_score_layer_calls", "metrics_path",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    extras = sorted({key for row in rows for key in row}.difference(fields))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + extras)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def summarize(rows):
    summaries = []
    for method, spec in METHODS.items():
        group = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["evaluation_seed"]),
        )
        if [int(row["evaluation_seed"]) for row in group] != list(SEEDS):
            raise ValueError(f"{method} does not have evaluation seeds 1, 2, 3")
        result = {
            "method": method, "label": spec["label"], "num_seeds": 3,
            "evaluation_seeds": "1,2,3", "rank_budget": 1536,
            "physical_rank": spec["physical_rank"],
            "evaluation_rng_mode": "per-episode",
            "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        }
        for metric in SUMMARY_METRICS:
            values = [finite_number(row.get(metric)) for row in group]
            if all(value is not None for value in values):
                result[f"{metric}_mean"] = statistics.mean(values)
                result[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(result)
    return summaries


def signature(args, paths):
    return {
        "checkpoints": {key: str(value) for key, value in paths.items()},
        "base_model_dir": str(args.base_model_dir.resolve()),
        "exp_pool_path": str(args.exp_pool_path.resolve()),
        "rank_budget": args.rank_budget, "trace": args.trace,
        "trace_num": args.trace_num, "video": args.video,
        "evaluation_seeds": list(SEEDS),
        "evaluation_rng_mode": "per-episode",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "selectors": "disabled", "nbs_compaction": False,
    }


def validate_or_write_manifest(args, paths):
    path = args.output_dir / "pipeline_manifest.json"
    expected = {"signature": signature(args, paths)}
    if path.is_file() and args.resume:
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("resume manifest differs from this configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    path.write_text(json.dumps(expected, indent=2), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    paths = checkpoint_paths(args)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / "lora_methods_per_seed.csv"

    if not args.dry_run:
        if args.rank_budget != 1536:
            raise ValueError("this comparison requires --rank-budget 1536")
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
        for method, path in paths.items():
            validate_checkpoint(path, method, args.rank_budget)
        validate_or_write_manifest(args, paths)

    rows = load_rows(runs_path) if args.resume else []
    completed = {
        (row["method"], int(row["evaluation_seed"])) for row in rows
    }
    for method, spec in METHODS.items():
        for seed in SEEDS:
            key = (method, seed)
            if key in completed:
                print(f"[{method} seed={seed}] already complete; skipping", flush=True)
                continue
            command = build_command(args, method, paths[method], seed)
            print(f"[{method} seed={seed}] {shlex.join(command)}", flush=True)
            if args.dry_run:
                continue
            started_at = time.time() - 1.0
            subprocess.run(command, cwd=ABR_ROOT, check=True)
            metrics_path = newest_metrics(started_at)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({
                "method": method, "label": spec["label"],
                "evaluation_seed": seed, "rank_budget": args.rank_budget,
                "physical_rank": spec["physical_rank"],
                "checkpoint_dir": str(paths[method]),
                "metrics_path": str(metrics_path.resolve()),
                **scalar_metrics(metrics),
            })
            write_rows(runs_path, rows)

    if args.dry_run:
        return
    summary_path = args.output_dir / "lora_methods_three_seed_summary.csv"
    write_rows(summary_path, summarize(rows))
    print(f"Per-seed results: {runs_path}")
    print(f"Three-seed summary: {summary_path}")


if __name__ == "__main__":
    main()
