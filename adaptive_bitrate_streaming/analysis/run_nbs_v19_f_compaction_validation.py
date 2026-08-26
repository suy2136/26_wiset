"""Compare dense and physically compacted inference for NBS v19 experiment F."""

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
if str(ABR_ROOT) not in sys.path:
    sys.path.insert(0, str(ABR_ROOT))

try:
    from adaptive_bitrate_streaming.analysis.run_nbs_v19_ef_inference import (
        DEFAULT_BASE_MODEL,
        DEFAULT_EXP_POOL,
        DEFAULT_TRAINING_STATE,
        RESULTS_ROOT,
        checkpoints_from_state,
        newest_metrics,
        scalar_metrics,
        validate_checkpoint,
    )
except ModuleNotFoundError:
    # Direct execution from the ABR repository root.
    from analysis.run_nbs_v19_ef_inference import (
        DEFAULT_BASE_MODEL,
        DEFAULT_EXP_POOL,
        DEFAULT_TRAINING_STATE,
        RESULTS_ROOT,
        checkpoints_from_state,
        newest_metrics,
        scalar_metrics,
        validate_checkpoint,
    )


EXPERIMENT = {"name": "F", "rank_budget": 3072, "mean_active_rank": 48.0}
DEFAULT_OUTPUT = RESULTS_ROOT / "nbs_v19_f_compaction_validation.csv"
RANK_CONFIG = "configs/nbs_v19_rank_config_max64.json"


def build_test_command(args, checkpoint, compact):
    return [
        sys.executable, "run_plm.py", "--test", "--nbs-v19", "--fp16",
        "--seed", "1", "--plm-type", "llama", "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--model-dir", str(checkpoint.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--rank", "64", "--nbs-rank-budget", "3072",
        "--nbs-rank-config", RANK_CONFIG,
        "--trace", args.trace, "--trace-num", str(args.trace_num),
        "--video", args.video, "--fixed-order",
        "--device", args.device, "--device-out", args.device,
        "--temporal-selector", "none", "--token-selector", "none",
        "--speculative-draft-steps", "0",
        "--nbs-compact-inference" if compact
        else "--no-nbs-compact-inference",
    ]


def load_rows(path):
    json_path = path.with_suffix(".json")
    if not json_path.is_file():
        return []
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    for row in rows:
        for field in ("mean_reward", "inference_latency_mean_ms"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"saved {row['mode']} {field} is non-finite")
    return rows


def write_rows(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "mode", "mean_reward", "inference_latency_mean_ms",
        "inference_latency_p50_ms", "inference_latency_p95_ms",
        "nbs_compact_inference", "nbs_physical_rank_total_before",
        "nbs_compact_rank_total", "nbs_compaction_rank_reduction_ratio",
        "nbs_compaction_logits_equivalent",
        "nbs_compaction_logits_max_abs_error",
        "checkpoint_dir", "metrics_path",
    ]
    fields = [field for field in preferred if field in fields] + [
        field for field in fields if field not in preferred
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )


def comparison_summary(rows, qoe_atol):
    by_mode = {row["mode"]: row for row in rows}
    if set(by_mode) != {"dense", "compact"}:
        return None
    dense, compact = by_mode["dense"], by_mode["compact"]
    qoe_delta = float(compact["mean_reward"]) - float(dense["mean_reward"])
    dense_latency = float(dense["inference_latency_mean_ms"])
    compact_latency = float(compact["inference_latency_mean_ms"])
    return {
        "checkpoint": dense["checkpoint_dir"],
        "qoe_dense": float(dense["mean_reward"]),
        "qoe_compact": float(compact["mean_reward"]),
        "qoe_delta": qoe_delta,
        "qoe_atol": qoe_atol,
        "qoe_equivalent": abs(qoe_delta) <= qoe_atol,
        "latency_dense_mean_ms": dense_latency,
        "latency_compact_mean_ms": compact_latency,
        "latency_reduction_ratio": 1.0 - compact_latency / dense_latency,
        "inference_speedup": dense_latency / compact_latency,
        "logits_equivalent": bool(
            compact.get("nbs_compaction_logits_equivalent", False)
        ),
        "logits_max_abs_error": compact.get(
            "nbs_compaction_logits_max_abs_error"
        ),
        "physical_rank_total_before": compact.get(
            "nbs_physical_rank_total_before"
        ),
        "compact_rank_total": compact.get("nbs_compact_rank_total"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-state", type=Path, default=DEFAULT_TRAINING_STATE)
    parser.add_argument("--f-checkpoint-dir", type=Path)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path, default=DEFAULT_EXP_POOL)
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qoe-atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.qoe_atol < 0:
        raise ValueError("--qoe-atol must be non-negative")
    checkpoint = (
        args.f_checkpoint_dir
        if args.f_checkpoint_dir is not None
        else checkpoints_from_state(args.training_state)["F"]
    )
    if not args.dry_run:
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
        validate_checkpoint(checkpoint, EXPERIMENT)

    rows = load_rows(args.output) if args.resume else []
    completed = {row["mode"] for row in rows}
    for mode, compact in (("dense", False), ("compact", True)):
        if mode in completed:
            print(f"[{mode}] already complete; skipping", flush=True)
            continue
        command = build_test_command(args, checkpoint, compact)
        print(f"[{mode}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        started_at = time.time() - 1.0
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = newest_metrics(started_at)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "mode": mode,
            "checkpoint_dir": str(checkpoint.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "rank_budget": 3072,
            "mean_active_rank": 48.0,
            "physical_rank": 64,
            "seed": 1,
            **scalar_metrics(metrics),
        })
        write_rows(rows, args.output)

    if args.dry_run:
        return
    summary = comparison_summary(rows, args.qoe_atol)
    if summary is None:
        raise RuntimeError("dense and compact F results are both required")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Results saved at: {args.output.resolve()}")
    if not summary["logits_equivalent"] or not summary["qoe_equivalent"]:
        raise RuntimeError(
            "F dense/compact equivalence failed; inspect "
            f"{summary_path.resolve()}"
        )


if __name__ == "__main__":
    main()
