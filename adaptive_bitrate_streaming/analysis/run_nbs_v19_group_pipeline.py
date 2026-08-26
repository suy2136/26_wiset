"""Shared sequential train-and-compact-test pipeline for ABR NBS sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ABR_ROOT / "data" / "ft_plms"
RESULTS_ROOT = ABR_ROOT / "artifacts" / "results"
DEFAULT_BASE_MODEL = ABR_ROOT.parent / "downloaded_plms" / "llama" / "base"
DEFAULT_EXP_POOL = ABR_ROOT / "artifacts" / "exp_pools" / "exp_pool.pkl"


def build_training_command(args, experiment):
    return [
        sys.executable, "run_plm.py", "--adapt", "--nbs-v19", "--fp16",
        "--seed", "1", "--plm-type", "llama", "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--rank", str(experiment["physical_rank"]),
        "--nbs-rank-budget", str(experiment["rank_budget"]),
        "--nbs-rank-config", experiment["rank_config"],
        "--lr", str(experiment["lr"]),
        "--lr-schedule", "cosine",
        "--warmup-steps", str(experiment["warmup_steps"]),
        "--num-epochs", str(args.num_epochs),
        "--eval-per-epoch", str(args.eval_per_epoch),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--early-stopping-min-epochs", str(args.early_stopping_min_epochs),
        "--early-stopping-min-delta", str(args.early_stopping_min_delta),
        "--plateau-lr-patience", str(args.plateau_lr_patience),
        "--plateau-lr-factor", str(args.plateau_lr_factor),
        "--plateau-min-lr", str(args.plateau_min_lr),
        "--trace", args.train_trace, "--trace-num", str(args.trace_num),
        "--video", args.video, "--fixed-order",
        "--device", args.device, "--device-out", args.device,
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--temporal-selector", "none", "--token-selector", "none",
        "--speculative-draft-steps", "0",
        "--save-checkpoint-per-epoch", "10",
        "--checkpoint-retention", "best-latest",
    ]


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
        "--lr", str(experiment["lr"]),
        "--lr-schedule", "cosine",
        "--warmup-steps", str(experiment["warmup_steps"]),
        "--num-epochs", str(args.num_epochs),
        "--trace", args.test_trace, "--trace-num", str(args.trace_num),
        "--video", args.video, "--fixed-order",
        "--device", args.device, "--device-out", args.device,
        "--temporal-selector", "none", "--token-selector", "none",
        "--speculative-draft-steps", "0",
        "--nbs-compact-inference",
    ]


def discover_best_checkpoint(experiment, started_at):
    marker = (
        f"rank_{experiment['physical_rank']}_nbs_v19_"
        f"budget{experiment['rank_budget']}_"
    )
    lr_marker = f"_lr_{experiment['lr']}_"
    candidates = []
    for metadata_path in MODEL_ROOT.rglob("checkpoint_metadata.json"):
        text = str(metadata_path)
        if (
            marker not in text
            or lr_marker not in text
            or metadata_path.stat().st_mtime < started_at
        ):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("role") == "best"
            and metadata.get("effective_rank_budget")
            == experiment["rank_budget"]
        ):
            candidates.append(metadata_path.parent)
    if not candidates:
        raise RuntimeError(
            f"{experiment['name']} produced no new matching best checkpoint"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


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
        raise ValueError(f"{experiment['name']} is not seed-1 NBS v19")
    if metadata.get("effective_rank_budget") != experiment["rank_budget"]:
        raise ValueError(f"{experiment['name']} rank budget mismatch")
    adapter = json.loads(
        (path / "adapter_config.json").read_text(encoding="utf-8")
    )
    physical_rank = int(adapter.get("init_r", adapter.get("r", -1)))
    if physical_rank != experiment["physical_rank"]:
        raise ValueError(f"{experiment['name']} physical rank mismatch")
    return metadata


def newest_metrics(started_at):
    candidates = [
        path for path in RESULTS_ROOT.rglob("selector_metrics.json")
        if path.stat().st_mtime >= started_at
    ]
    if not candidates:
        raise RuntimeError("compact inference produced no selector_metrics.json")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalar_metrics(metrics):
    return {
        key: value for key, value in metrics.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def signature(args, experiments):
    return {
        "experiments": list(experiments),
        "base_model_dir": str(args.base_model_dir.resolve()),
        "exp_pool_path": str(args.exp_pool_path.resolve()),
        "train_trace": args.train_trace,
        "test_trace": args.test_trace,
        "trace_num": args.trace_num,
        "video": args.video,
        "device": args.device,
        "seed": 1,
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "min_epochs": args.early_stopping_min_epochs,
            "min_delta": args.early_stopping_min_delta,
        },
        "features": {
            "temporal_selector": "none", "token_selector": "none",
            "speculative_draft_steps": 0, "nbs_compact_inference": True,
        },
    }


def load_state(path, resume, run_signature):
    if not resume or not path.is_file():
        return {"signature": run_signature, "runs": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("signature") != run_signature:
        raise ValueError("saved group state does not match current experiment setup")
    return state


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def result_rows(state, experiments):
    rows = []
    for experiment in experiments:
        run = state["runs"].get(experiment["name"], {})
        metrics = run.get("metrics")
        if metrics is None:
            continue
        rows.append({
            "experiment": experiment["name"],
            "rank_budget": experiment["rank_budget"],
            "mean_active_rank": experiment["rank_budget"] / 64,
            "physical_rank": experiment["physical_rank"],
            "learning_rate": experiment["lr"],
            "lr_schedule": "cosine",
            "warmup_steps": experiment["warmup_steps"],
            "seed": 1,
            "checkpoint_dir": run["checkpoint_dir"],
            "metrics_path": run["metrics_path"],
            "compaction_equivalence_path": run.get(
                "compaction_equivalence_path"
            ),
            **metrics,
        })
    if rows:
        reference_qoe = float(rows[0]["mean_reward"])
        reference_latency = float(rows[0]["inference_latency_mean_ms"])
        for row in rows:
            row["mean_reward_delta_vs_first"] = (
                float(row["mean_reward"]) - reference_qoe
            )
            row["latency_reduction_vs_first"] = (
                1.0 - float(row["inference_latency_mean_ms"])
                / reference_latency
            )
    return rows


def write_results(path, rows, run_signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({key for row in rows for key in row})
        preferred = [
            "experiment", "rank_budget", "mean_active_rank",
            "physical_rank", "learning_rate", "lr_schedule", "warmup_steps",
            "mean_reward", "mean_reward_delta_vs_first",
            "inference_latency_mean_ms", "latency_reduction_vs_first",
            "nbs_compact_inference", "nbs_compaction_logits_equivalent",
            "nbs_compaction_rank_reduction_ratio", "token_reduction_ratio",
            "checkpoint_dir", "metrics_path",
        ]
        fields = [field for field in preferred if field in fields] + [
            field for field in fields if field not in preferred
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(path.with_suffix(".json"), rows)
    atomic_json(path.with_suffix(".manifest.json"), run_signature)


def parse_args(argv, state_file, output_file):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path, default=DEFAULT_EXP_POOL)
    parser.add_argument("--train-trace", default="fcc-valid")
    parser.add_argument("--test-trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--eval-per-epoch", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.003)
    parser.add_argument("--plateau-lr-patience", type=int, default=5)
    parser.add_argument("--plateau-lr-factor", type=float, default=0.5)
    parser.add_argument("--plateau-min-lr", type=float, default=1e-6)
    parser.add_argument("--state-file", type=Path, default=state_file)
    parser.add_argument("--output", type=Path, default=output_file)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run_group(args, experiments):
    run_signature = signature(args, experiments)
    if not args.dry_run:
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
    state = load_state(args.state_file, args.resume, run_signature)
    metrics_dir = args.output.parent / f"{args.output.stem}_metrics"

    for experiment in experiments:
        name = experiment["name"]
        run = state["runs"].setdefault(name, {})
        checkpoint_text = run.get("checkpoint_dir")
        if checkpoint_text is None:
            train_command = build_training_command(args, experiment)
            print(f"[{name}:train] {shlex.join(train_command)}", flush=True)
            if args.dry_run:
                checkpoint = Path(f"/best_checkpoint/{name}")
            else:
                started_at = time.time() - 1.0
                subprocess.run(train_command, cwd=ABR_ROOT, check=True)
                checkpoint = discover_best_checkpoint(experiment, started_at)
                run.update({
                    "status": "trained",
                    "checkpoint_dir": str(checkpoint.resolve()),
                    "trained_at": time.time(),
                })
                atomic_json(args.state_file, state)
                print(f"[{name}] best checkpoint: {checkpoint}", flush=True)
        else:
            checkpoint = Path(checkpoint_text)
            print(f"[{name}:train] checkpoint already available; skipping", flush=True)

        if run.get("status") == "complete":
            print(f"[{name}:test] already complete; skipping", flush=True)
            continue
        test_command = build_test_command(args, experiment, checkpoint)
        print(f"[{name}:test] {shlex.join(test_command)}", flush=True)
        if args.dry_run:
            continue

        metadata = validate_checkpoint(checkpoint, experiment)
        started_at = time.time() - 1.0
        subprocess.run(test_command, cwd=ABR_ROOT, check=True)
        source_metrics = newest_metrics(started_at)
        metrics = json.loads(source_metrics.read_text(encoding="utf-8"))
        if not metrics.get("nbs_compact_inference"):
            raise RuntimeError(f"{name} inference did not use NBS compaction")
        if not metrics.get("nbs_compaction_logits_equivalent"):
            raise RuntimeError(f"{name} compact logits equivalence failed")

        metrics_dir.mkdir(parents=True, exist_ok=True)
        saved_metrics = metrics_dir / f"{name}_selector_metrics.json"
        shutil.copy2(source_metrics, saved_metrics)
        equivalence_source = (
            source_metrics.parent / "nbs_compaction_equivalence.json"
        )
        equivalence_path = None
        if equivalence_source.is_file():
            equivalence_path = metrics_dir / f"{name}_compaction_equivalence.json"
            shutil.copy2(equivalence_source, equivalence_path)
        run.update({
            "status": "complete",
            "checkpoint_role": metadata.get("role"),
            "metrics": scalar_metrics(metrics),
            "metrics_path": str(saved_metrics.resolve()),
            "compaction_equivalence_path": (
                None if equivalence_path is None
                else str(equivalence_path.resolve())
            ),
            "completed_at": time.time(),
        })
        atomic_json(args.state_file, state)
        write_results(
            args.output, result_rows(state, experiments), run_signature
        )
        print(
            f"[{name}] compact test QoE={metrics['mean_reward']:.6f} "
            f"latency={metrics['inference_latency_mean_ms']:.3f} ms",
            flush=True,
        )

    if not args.dry_run:
        rows = result_rows(state, experiments)
        write_results(args.output, rows, run_signature)
        print(f"Results saved at: {args.output.resolve()}")
