"""Train and evaluate four additive ABR NBS rank-bound ablations.

Every run uses a new run tag and output directory. Existing checkpoints and
results are only scanned for newly-created matching metadata and are never
modified. Each trained checkpoint is compact-evaluated with evaluation seeds
1, 2, and 3 using per-episode reseeding and FP16 prescaled Q/K attention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

try:
    from adaptive_bitrate_streaming.analysis import (
        run_nbs_v19_group_pipeline as training,
    )
except ModuleNotFoundError:
    import run_nbs_v19_group_pipeline as training


ABR_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ABR_ROOT / "data/ft_plms"
DEFAULT_OUTPUT = (
    ABR_ROOT / "artifacts/results/abr_nbs_rank_bound_ablation_c1536_data4"
)
RANK_BUDGET = 1536
PHYSICAL_RANK = 32
MODULE_COUNT = 64
EVALUATION_SEEDS = (1, 2, 3)
SPECS = (
    {
        "name": "min8_max32", "min_rank": 8, "max_rank": 32,
        "rank_config": "configs/nbs_v19_rank_config_min8_max32.json",
    },
    {
        "name": "min16_max32", "min_rank": 16, "max_rank": 32,
        "rank_config": "configs/nbs_v19_rank_config_min16_max32.json",
    },
    {
        "name": "min2_max28", "min_rank": 2, "max_rank": 28,
        "rank_config": "configs/nbs_v19_rank_config_min2_max28.json",
    },
    {
        "name": "min2_max24", "min_rank": 2, "max_rank": 24,
        "rank_config": "configs/nbs_v19_rank_config_min2_max24.json",
    },
)
SUMMARY_METRICS = (
    "mean_reward", "qoe_raw_mean", "mean_bitrate_mbps",
    "mean_rebuffer_s_per_chunk", "total_rebuffer_s",
    "mean_smoothness_mbps", "inference_latency_mean_ms",
    "inference_latency_p50_ms", "inference_latency_p95_ms",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data-seed", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-model-dir", type=Path, default=training.DEFAULT_BASE_MODEL,
    )
    parser.add_argument(
        "--exp-pool-path", type=Path, default=training.DEFAULT_EXP_POOL,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def experiment_for(args, spec):
    return {
        **spec,
        "method": "nbs",
        "rank_budget": RANK_BUDGET,
        "physical_rank": PHYSICAL_RANK,
        "lr": 2e-4,
        "warmup_steps": 500,
        "seed": 1,
        "lora_seed": 1,
        "data_seed": args.training_data_seed,
        "run_tag": (
            f"rank_bounds_c1536_data{args.training_data_seed}_{spec['name']}_"
            f"{safe_tag(args.output_dir.name)}"
        ),
        "nbs_allocation_audit": True,
    }


def training_args(args, spec):
    root = args.output_dir / "training" / spec["name"]
    argv = [
        "--base-model-dir", str(args.base_model_dir),
        "--exp-pool-path", str(args.exp_pool_path),
        "--device", args.device,
        "--state-file", str(root / "state.json"),
        "--output", str(root / "result.csv"),
        "--num-epochs", "80",
        "--early-stopping-min-epochs", "20",
        "--early-stopping-patience", "10",
        "--fp16-numeric-safeguards",
        "--fp16-selective-clamp",
        "--fp16-selective-clamp-threshold", "60000.0",
        "--skip-nonfinite-batches",
        "--nbs-skip-batch-at-rollback-lr-floor",
        "--nbs-rollback-min-lr", "1e-5",
        "--continue-on-error",
    ]
    if args.resume:
        argv.append("--resume")
    parsed = training.parse_args(
        argv, state_file=root / "state.json", output_file=root / "result.csv",
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    return parsed


def add_prescaled_qk(command):
    command = [item for item in command if item != "--fp16-attention-fp32-scores"]
    if "--fp16-attention-prescaled-qk" not in command:
        command.append("--fp16-attention-prescaled-qk")
    return command


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(args):
    return {
        "pipeline": "abr_nbs_rank_bound_ablation_v1",
        "rank_budget": RANK_BUDGET,
        "physical_rank": PHYSICAL_RANK,
        "training_seed": 1,
        "lora_seed": 1,
        "training_data_seed": args.training_data_seed,
        "evaluation_seeds_and_data_seeds": list(EVALUATION_SEEDS),
        "evaluation_rng_mode": "per-episode",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "compact_inference": True,
        "specs": list(SPECS),
        "base_model_dir": str(args.base_model_dir.resolve()),
        "exp_pool_path": str(args.exp_pool_path.resolve()),
        "trace": args.trace,
        "trace_num": args.trace_num,
        "video": args.video,
        "device": args.device,
    }


def load_state(args):
    path = args.output_dir / "pipeline_state.json"
    expected = signature(args)
    if path.is_file():
        if not args.resume:
            raise FileExistsError(f"output exists: {path}; use --resume")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("signature") != expected:
            raise ValueError("resume state differs from the current configuration")
        return path, state
    state = {"signature": expected, "runs": {}}
    if not args.dry_run:
        atomic_json(path, state)
    return path, state


def discover_checkpoint(experiment, started_at=0.0):
    candidates = []
    for metadata_path in MODEL_ROOT.rglob("checkpoint_metadata.json"):
        if metadata_path.stat().st_mtime < started_at:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            metadata.get("variant") == "nbs_v19"
            and metadata.get("role") == "best"
            and int(metadata.get("seed", -1)) == 1
            and int(metadata.get("lora_seed", metadata.get("seed", -1))) == 1
            and int(metadata.get("data_seed", metadata.get("seed", -1)))
            == experiment["data_seed"]
            and int(metadata.get("physical_rank", -1)) == PHYSICAL_RANK
            and int(metadata.get("effective_rank_budget", -1)) == RANK_BUDGET
            and metadata.get("run_tag") == experiment["run_tag"]
        ):
            candidates.append(metadata_path.parent)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def validate_checkpoint(path: Path, experiment):
    required = (
        "adapter_config.json", "modules_except_plm.bin",
        "checkpoint_metadata.json", "nash_rank_allocator.pt",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if not any((path / name).is_file() for name in (
        "adapter_model.bin", "adapter_model.safetensors",
    )):
        missing.append("adapter weights")
    if missing:
        raise FileNotFoundError(f"incomplete NBS checkpoint: {missing}")
    metadata = json.loads(
        (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    expected = {
        "variant": "nbs_v19", "role": "best", "seed": 1,
        "lora_seed": 1, "data_seed": experiment["data_seed"],
        "physical_rank": PHYSICAL_RANK,
        "effective_rank_budget": RANK_BUDGET,
        "run_tag": experiment["run_tag"],
    }
    mismatches = {
        key: (metadata.get(key, metadata.get("seed")), value)
        for key, value in expected.items()
        if metadata.get(key, metadata.get("seed")) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint identity mismatch: {mismatches}")
    return metadata


def train_one(args, spec, run, state, state_path):
    experiment = experiment_for(args, spec)
    saved = run.get("checkpoint_dir")
    if saved:
        checkpoint = Path(saved)
        validate_checkpoint(checkpoint, experiment)
        print(f"[{spec['name']}:train] checkpoint already recorded; skipping")
        return checkpoint
    if args.resume:
        checkpoint = discover_checkpoint(experiment)
        if checkpoint is not None:
            validate_checkpoint(checkpoint, experiment)
            run["checkpoint_dir"] = str(checkpoint.resolve())
            atomic_json(state_path, state)
            print(f"[{spec['name']}:train] recovered checkpoint: {checkpoint}")
            return checkpoint
    command = add_prescaled_qk(
        training.build_training_command(training_args(args, spec), experiment)
    )
    print(f"[{spec['name']}:train] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return Path(f"/dry-run/{spec['name']}")
    started_at = time.time() - 1.0
    subprocess.run(command, cwd=ABR_ROOT, check=True)
    checkpoint = discover_checkpoint(experiment, started_at)
    if checkpoint is None:
        raise FileNotFoundError(f"new checkpoint not found: {spec['name']}")
    validate_checkpoint(checkpoint, experiment)
    run["checkpoint_dir"] = str(checkpoint.resolve())
    atomic_json(state_path, state)
    return checkpoint


def newest_metrics(started_at):
    candidates = [
        path for path in training.RESULTS_ROOT.rglob("selector_metrics.json")
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


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = [
        "experiment", "min_rank", "max_rank", "evaluation_seed",
        "training_data_seed", "rank_budget", "physical_rank",
        "mean_reward", "inference_latency_mean_ms",
        "final_rank_min", "final_rank_max", "final_rank_total",
        "attention_score_mode", "checkpoint_dir", "metrics_path",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    fields += sorted({key for row in rows for key in row}.difference(fields))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def rank_summary(metadata):
    ranks = metadata.get("active_ranks") or {}
    values = [int(value) for value in ranks.values()] if isinstance(ranks, dict) else []
    return {
        "final_rank_min": min(values) if values else None,
        "final_rank_max": max(values) if values else None,
        "final_rank_total": sum(values) if values else RANK_BUDGET,
    }


def evaluate_one(args, spec, checkpoint, rows, output):
    experiment = experiment_for(args, spec)
    metadata = {} if args.dry_run else validate_checkpoint(checkpoint, experiment)
    ranks = rank_summary(metadata)
    completed = {
        int(row["evaluation_seed"])
        for row in rows if row.get("experiment") == spec["name"]
        and row.get("checkpoint_dir") == str(checkpoint.resolve())
        and row.get("status") == "complete"
    }
    for seed in EVALUATION_SEEDS:
        if seed in completed:
            print(f"[{spec['name']} seed={seed}] already complete; skipping")
            continue
        evaluation = {
            **experiment, "seed": seed, "lora_seed": 1, "data_seed": seed,
            "run_tag": f"{experiment['run_tag']}_eval{seed}",
        }
        command = add_prescaled_qk(
            training.build_test_command(
                training_args(args, spec), evaluation, checkpoint,
            )
        )
        print(f"[{spec['name']} seed={seed}:test] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        row = {
            "experiment": spec["name"], "min_rank": spec["min_rank"],
            "max_rank": spec["max_rank"], "evaluation_seed": seed,
            "training_data_seed": args.training_data_seed,
            "rank_budget": RANK_BUDGET, "physical_rank": PHYSICAL_RANK,
            "rank_config": spec["rank_config"],
            "checkpoint_dir": str(checkpoint.resolve()),
            "evaluation_rng_mode": "per-episode",
            "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
            "status": "failed", **ranks,
        }
        try:
            started_at = time.time() - 1.0
            subprocess.run(command, cwd=ABR_ROOT, check=True)
            metrics_path = newest_metrics(started_at)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row.update(scalar_metrics(metrics))
            row["metrics_path"] = str(metrics_path.resolve())
            row["status"] = "complete"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        write_rows(output, rows)
        if row["status"] != "complete":
            raise RuntimeError(row["error"])
    return rows


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize(rows):
    result = []
    for spec in SPECS:
        group = sorted(
            (row for row in rows if row.get("experiment") == spec["name"]
             and row.get("status") == "complete"),
            key=lambda row: int(row["evaluation_seed"]),
        )
        if [int(row["evaluation_seed"]) for row in group] != list(EVALUATION_SEEDS):
            continue
        summary = {
            "experiment": spec["name"], "min_rank": spec["min_rank"],
            "max_rank": spec["max_rank"], "num_evaluation_seeds": 3,
            "evaluation_seeds": "1,2,3", "rank_budget": RANK_BUDGET,
            "physical_rank": PHYSICAL_RANK,
            "training_data_seed": group[0]["training_data_seed"],
            "final_rank_min": group[0].get("final_rank_min"),
            "final_rank_max": group[0].get("final_rank_max"),
            "final_rank_total": group[0].get("final_rank_total"),
            "evaluation_rng_mode": "per-episode",
            "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
            "checkpoint_dir": group[0]["checkpoint_dir"],
        }
        for metric in SUMMARY_METRICS:
            values = [finite(row.get(metric)) for row in group]
            if all(value is not None for value in values):
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values)
        result.append(summary)
    return result


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.base_model_dir = args.base_model_dir.resolve()
    args.exp_pool_path = args.exp_pool_path.resolve()
    if not args.dry_run:
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        minimum = spec["min_rank"] * MODULE_COUNT
        maximum = spec["max_rank"] * MODULE_COUNT
        if not minimum <= RANK_BUDGET <= maximum:
            raise ValueError(
                f"infeasible bounds for {spec['name']}: "
                f"budget {RANK_BUDGET} is outside [{minimum}, {maximum}]"
            )
    state_path, state = load_state(args)
    output = args.output_dir / "per_seed_results.csv"
    rows = load_rows(output) if args.resume else []

    for spec in SPECS:
        run = state["runs"].setdefault(spec["name"], {})
        try:
            checkpoint = train_one(args, spec, run, state, state_path)
            rows = evaluate_one(args, spec, checkpoint, rows, output)
        except Exception as error:
            run.update({
                "status": "failed", "error_type": type(error).__name__,
                "error": str(error), "finished_at": time.time(),
            })
            print(
                f"[{spec['name']}] FAILED: {type(error).__name__}: {error}",
                file=sys.stderr, flush=True,
            )
        else:
            if not args.dry_run:
                run.update({"status": "complete", "finished_at": time.time()})
        if not args.dry_run:
            atomic_json(state_path, state)

    if args.dry_run:
        return
    summary_path = args.output_dir / "three_seed_summary.csv"
    write_rows(summary_path, summarize(rows))
    failed = [name for name, run in state["runs"].items() if run.get("status") == "failed"]
    atomic_json(args.output_dir / "final_report.json", {
        "runs": state["runs"], "failed": failed,
        "per_seed_results": str(output), "summary": str(summary_path),
    })
    print(f"Per-seed results: {output}")
    print(f"Three-seed summary: {summary_path}")
    print(f"Failed experiments: {failed or 'none'}")
    print(f"OUTPUT={args.output_dir}")


if __name__ == "__main__":
    main()
