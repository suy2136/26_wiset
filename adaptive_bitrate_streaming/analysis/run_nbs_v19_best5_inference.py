"""Evaluate five isolated/combined ABR inference configurations.

All runs reuse one NBS-v19 budget-1536 checkpoint and enable physical NBS
compaction.  No training is performed.  Results are checkpointed after every
completed evaluation so ``--resume`` can safely continue an interrupted run.
"""

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
DEFAULT_OUTPUT = RESULTS_ROOT / "nbs_v19_budget1536_best5_inference.csv"

EXPERIMENTS = (
    {"name": "nbs_compact_only"},
    {"name": "temporal_k1", "temporal": True},
    {"name": "token_offsets_2_6_7", "token": True},
    {"name": "spec_repeat_last_k3", "speculative": True},
    {
        "name": "combined_temporal_token_spec",
        "temporal": True,
        "token": True,
        "speculative": True,
    },
)


def validate_checkpoint(checkpoint_dir, rank_budget):
    required = (
        "adapter_config.json", "modules_except_plm.bin",
        "nash_rank_allocator.pt", "checkpoint_metadata.json",
    )
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if not any(
        (checkpoint_dir / name).is_file()
        for name in ("adapter_model.bin", "adapter_model.safetensors")
    ):
        missing.append("adapter_model.bin or adapter_model.safetensors")
    if missing:
        raise FileNotFoundError(f"incomplete NBS checkpoint: {', '.join(missing)}")
    metadata = json.loads(
        (checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("variant") != "nbs_v19" or metadata.get("seed") != 1:
        raise ValueError("checkpoint must be NBS v19 seed 1")
    if metadata.get("effective_rank_budget") != rank_budget:
        raise ValueError("checkpoint rank budget does not match --rank-budget")
    return metadata


def build_command(args, experiment):
    temporal = bool(experiment.get("temporal"))
    token = bool(experiment.get("token"))
    speculative = bool(experiment.get("speculative"))
    command = [
        sys.executable, "run_plm.py", "--test", "--nbs-v19", "--fp16",
        "--seed", str(args.data_seed), "--lora-seed", "1",
        "--data-seed", str(args.data_seed),
        "--plm-type", "llama", "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--model-dir", str(args.checkpoint_dir.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--rank", str(args.physical_rank),
        "--nbs-rank-budget", str(args.rank_budget),
        "--nbs-rank-config", str(args.rank_config),
        "--trace", args.trace, "--trace-num", str(args.trace_num),
        "--video", args.video, "--fixed-order",
        "--device", args.device, "--device-out", args.device,
        "--nbs-compact-inference",
        "--temporal-selector", "event-aware" if temporal else "none",
        "--token-selector", "intra-timestep" if token else "none",
        "--selector-history-steps", "20",
        "--speculative-draft-steps", "3" if speculative else "0",
        "--speculative-drafter", "repeat-last" if speculative else "mpc",
        "--speculative-verification-mode", "sample",
        "--speculative-buffer-tolerance", "1.0",
        "--speculative-state-tolerance", "0.25",
        "--speculative-return-tolerance", "0.01",
    ]
    if temporal:
        command.extend([
            "--event-max-events", "1", "--event-min-spacing", "2",
            "--event-throughput-threshold", "0.6",
            "--event-buffer-threshold", "6.0",
            "--event-bitrate-jump-threshold", "1",
        ])
    if token:
        command.extend(["--intra-token-keep-offsets", "2", "6", "7"])
    return command


def newest_metrics(started_at):
    candidates = [
        path for path in RESULTS_ROOT.rglob("selector_metrics.json")
        if path.stat().st_mtime >= started_at
    ]
    if not candidates:
        raise RuntimeError("evaluation produced no selector_metrics.json")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalar_metrics(metrics):
    return {
        key: value for key, value in metrics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def add_baseline_comparisons(rows):
    baseline = next(
        (row for row in rows if row["experiment"] == "nbs_compact_only"), None
    )
    if baseline is None:
        return rows
    base_reward = baseline.get("mean_reward")
    base_latency = baseline.get("inference_latency_mean_ms")
    for row in rows:
        reward = row.get("mean_reward")
        latency = row.get("inference_latency_mean_ms")
        if all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in (reward, base_reward)):
            row["mean_reward_delta_vs_nbs"] = reward - base_reward
            row["mean_reward_change_ratio_vs_nbs"] = (
                0.0 if base_reward == 0 else reward / base_reward - 1.0
            )
        if all(isinstance(value, (int, float)) and value > 0
               for value in (latency, base_latency)):
            row["inference_speedup_vs_nbs"] = base_latency / latency
            row["inference_latency_reduction_vs_nbs"] = 1.0 - latency / base_latency
    return rows


def write_results(rows, output, signature):
    add_baseline_comparisons(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "experiment", "mean_reward", "mean_reward_delta_vs_nbs",
        "mean_reward_change_ratio_vs_nbs", "qoe_raw_mean",
        "mean_bitrate_mbps", "total_rebuffer_s", "mean_rebuffer_s_per_chunk",
        "mean_smoothness_mbps", "inference_latency_mean_ms",
        "inference_latency_p50_ms", "inference_latency_p95_ms",
        "inference_speedup_vs_nbs", "inference_latency_reduction_vs_nbs",
        "token_reduction_ratio", "temporal_history_reduction_ratio",
        "intra_token_reduction_ratio", "llm_call_reduction_ratio",
        "acceptance_rate", "target_plm_calls", "metrics_path",
        "configured_temporal", "configured_token",
        "configured_speculative", "configured_token_offsets",
        "configured_speculative_drafter",
    ]
    all_fields = {key for row in rows for key in row}
    fields = [key for key in preferred if key in all_fields]
    fields.extend(sorted(all_fields.difference(fields)))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps({
            "signature": signature,
            "completed_experiments": [row["experiment"] for row in rows],
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_resume(output, signature):
    rows_path = output.with_suffix(".json")
    manifest_path = output.with_suffix(".manifest.json")
    if not rows_path.exists() and not manifest_path.exists():
        return []
    if not rows_path.exists() or not manifest_path.exists():
        raise RuntimeError("resume requires both result and manifest JSON")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("signature") != signature:
        raise ValueError("existing manifest does not match this run")
    return json.loads(rows_path.read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path, default=DEFAULT_EXP_POOL)
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
        "--data-seed", type=int, default=1,
        help=(
            "evaluation replicate seed; controls inference randomness and is "
            "also recorded as the data seed (the checkpoint LoRA seed stays 1)"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only", nargs="*", choices=[item["name"] for item in EXPERIMENTS]
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    selected = [
        item for item in EXPERIMENTS
        if not args.only or item["name"] in args.only
    ]
    if args.dry_run:
        for experiment in selected:
            print(
                f"[{experiment['name']}] "
                f"{shlex.join(build_command(args, experiment))}",
                flush=True,
            )
        return

    validate_checkpoint(args.checkpoint_dir, args.rank_budget)
    if not (args.base_model_dir / "config.json").is_file():
        raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
    if not args.exp_pool_path.is_file():
        raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
    signature = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "base_model_dir": str(args.base_model_dir.resolve()),
        "exp_pool_path": str(args.exp_pool_path.resolve()),
        "rank_budget": args.rank_budget, "physical_rank": args.physical_rank,
        "rank_config": str(args.rank_config), "seed": args.data_seed,
        "lora_seed": 1, "data_seed": args.data_seed,
        "trace": args.trace, "trace_num": args.trace_num, "video": args.video,
        "experiments": [item["name"] for item in EXPERIMENTS],
    }
    rows = load_resume(args.output, signature) if args.resume else []
    completed = {row["experiment"] for row in rows}
    for experiment in selected:
        name = experiment["name"]
        if name in completed:
            print(f"[{name}] already complete; skipping", flush=True)
            continue
        command = build_command(args, experiment)
        print(f"[{name}] {shlex.join(command)}", flush=True)
        started_at = time.time() - 1.0
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = newest_metrics(started_at)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "experiment": name,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "rank_budget": args.rank_budget,
            "physical_rank": args.physical_rank,
            "data_seed": args.data_seed,
            "configured_temporal": bool(experiment.get("temporal")),
            "configured_token": bool(experiment.get("token")),
            "configured_speculative": bool(experiment.get("speculative")),
            "configured_token_offsets": (
                "2,6,7" if experiment.get("token") else None
            ),
            "configured_speculative_drafter": (
                "repeat-last" if experiment.get("speculative") else None
            ),
            "metrics_path": str(metrics_path.resolve()),
            **scalar_metrics(metrics),
        })
        write_results(rows, args.output, signature)
    print(f"Results saved at: {args.output.resolve()}")


if __name__ == "__main__":
    main()
