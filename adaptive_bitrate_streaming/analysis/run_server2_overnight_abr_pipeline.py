"""Run the ordered, restartable ABR workload assigned to server 2.

The pipeline never edits an existing checkpoint.  Newly trained EVA data-seed-1
and Uniform data-seed-3 adapters use unique run tags and best/latest retention.
Existing NBS-v19 and AdaLoRA data-seed-3 checkpoints are discovered from their
metadata and opened read-only.  Every reported evaluation uses three evaluation
seeds, per-episode RNG reseeding, and FP16 pre-scaled Q/K attention.
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

try:
    from adaptive_bitrate_streaming.analysis import (
        run_abr_lora_methods_fp16_prescaled_multiseed as lora_eval,
    )
    from adaptive_bitrate_streaming.analysis import (
        run_nbs_v19_group_pipeline as training,
    )
    from adaptive_bitrate_streaming.analysis import (
        run_nbs_v19_module_sweep_pipeline as module_sweep,
    )
    from adaptive_bitrate_streaming.analysis import (
        run_nbs_v19_reseed_latest_modules_multiseed as latest_modules,
    )
except ModuleNotFoundError:
    import run_abr_lora_methods_fp16_prescaled_multiseed as lora_eval
    import run_nbs_v19_group_pipeline as training
    import run_nbs_v19_module_sweep_pipeline as module_sweep
    import run_nbs_v19_reseed_latest_modules_multiseed as latest_modules


ABR_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ABR_ROOT / "data" / "ft_plms"
DEFAULT_OUTPUT = (
    ABR_ROOT / "artifacts" / "results" / "server2_overnight_abr_pipeline"
)
SEEDS = (1, 2, 3)

EVA_DATA1 = {
    "name": "EVA_C1536_DATA1_SERVER2_NIGHT",
    "method": "eva",
    "rank_budget": 1536,
    "physical_rank": 32,
    "lr": 2e-4,
    "warmup_steps": 500,
    "seed": 1,
    "lora_seed": 1,
    "data_seed": 1,
    "run_tag": "server2_night_eva_data1_v1",
    "min_rank": 2,
    "max_rank": 32,
    "eva_metric": "ratio",
    "eva_similarity_threshold": 0.99,
    "eva_min_batches": 2,
    "eva_max_batches": 512,
    "eva_allow_unconverged": True,
}

UNIFORM_DATA3 = {
    "name": "UNIFORM_R24_DATA3_SERVER2_NIGHT",
    "method": "uniform_lora",
    "rank_budget": 1536,
    "physical_rank": 24,
    "lr": 2e-4,
    "warmup_steps": 500,
    "seed": 1,
    "lora_seed": 1,
    "data_seed": 3,
    "run_tag": "server2_night_uniform_data3_v1",
}

STAGE_ORDER = (
    "train_eva_data1",
    "evaluate_eva_data1",
    "validate_existing_data3",
    "evaluate_adalora_data3",
    "evaluate_nbs_data3_modules",
    "train_uniform_data3",
    "evaluate_uniform_data3",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model-dir", type=Path,
                        default=training.DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path,
                        default=training.DEFAULT_EXP_POOL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(args):
    return {
        "pipeline": "server2_overnight_abr_v1",
        "stage_order": list(STAGE_ORDER),
        "base_model_dir": str(args.base_model_dir.resolve()),
        "exp_pool_path": str(args.exp_pool_path.resolve()),
        "device": args.device,
        "trace": args.trace,
        "trace_num": args.trace_num,
        "video": args.video,
        "evaluation_seeds": list(SEEDS),
        "evaluation_rng_mode": "per-episode",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "new_training": [EVA_DATA1, UNIFORM_DATA3],
    }


def load_state(args):
    path = args.output_dir / "pipeline_state.json"
    expected = signature(args)
    if path.is_file():
        if not args.resume:
            raise FileExistsError(
                f"output already exists: {args.output_dir}; use --resume"
            )
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("signature") != expected:
            raise ValueError("resume state differs from this configuration")
        return path, state
    state = {"signature": expected, "stages": {}, "checkpoints": {}}
    if not args.dry_run:
        atomic_json(path, state)
    return path, state


def training_args(args, experiment, stem):
    argv = [
        "--base-model-dir", str(args.base_model_dir),
        "--exp-pool-path", str(args.exp_pool_path),
        "--device", args.device,
        "--state-file", str(args.output_dir / "training" / f"{stem}_state.json"),
        "--output", str(args.output_dir / "training" / f"{stem}_result.csv"),
        "--fp16-numeric-safeguards",
        "--fp16-selective-clamp",
        "--fp16-selective-clamp-threshold", "60000.0",
        "--skip-nonfinite-batches",
        "--nbs-skip-batch-at-rollback-lr-floor",
        "--nbs-rollback-min-lr", "1e-5",
        "--continue-on-error",
    ]
    parsed = training.parse_args(
        argv,
        state_file=args.output_dir / "training" / f"{stem}_state.json",
        output_file=args.output_dir / "training" / f"{stem}_result.csv",
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    return parsed


def train_new_checkpoint(args, state, state_path, key, experiment):
    saved = state["checkpoints"].get(key)
    if saved:
        checkpoint = Path(saved)
        training.validate_checkpoint(checkpoint, experiment)
        print(f"[{key}] validated existing pipeline checkpoint: {checkpoint}", flush=True)
        return checkpoint

    train_args = training_args(args, experiment, key)
    if training.experiment_method(experiment) == "eva":
        eva_state = training.eva_state_dir(train_args, experiment) / "eva_state.pt"
        if not eva_state.is_file():
            command = training.build_eva_precompute_command(train_args, experiment)
            print(f"[{key}:eva] {shlex.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ABR_ROOT, check=True)

    command = training.build_training_command(train_args, experiment)
    print(f"[{key}:train] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return Path(f"/dry-run/{key}")
    started_at = time.time() - 1.0
    subprocess.run(command, cwd=ABR_ROOT, check=True)
    checkpoint = training.discover_best_checkpoint(experiment, started_at)
    training.validate_checkpoint(checkpoint, experiment)
    state["checkpoints"][key] = str(checkpoint.resolve())
    atomic_json(state_path, state)
    print(f"[{key}] new checkpoint: {checkpoint}", flush=True)
    return checkpoint


def metadata(path):
    return json.loads(
        (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )


def discover_existing(method, data_seed):
    variant = {
        "nbs": "nbs_v19",
        "adalora": "adalora",
    }[method]
    candidates = []
    for path in MODEL_ROOT.rglob("checkpoint_metadata.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            item.get("variant") == variant
            and int(item.get("data_seed", item.get("seed", -1))) == data_seed
            and int(item.get("effective_rank_budget", -1)) == 1536
            and int(item.get("physical_rank", -1)) == 32
        ):
            candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(
            f"no {method} data-seed-{data_seed} budget-1536 checkpoint"
        )
    role_order = {"final": 2, "best": 1}
    return max(
        candidates,
        key=lambda path: (
            role_order.get(metadata(path).get("role"), 0),
            path.stat().st_mtime,
        ),
    )


def validate_existing(path, method, data_seed):
    required = (
        "adapter_config.json", "modules_except_plm.bin",
        "checkpoint_metadata.json",
    )
    if method == "nbs":
        required += ("nash_rank_allocator.pt",)
    missing = [name for name in required if not (path / name).is_file()]
    if not any((path / name).is_file() for name in (
        "adapter_model.bin", "adapter_model.safetensors",
    )):
        missing.append("adapter_model.bin or adapter_model.safetensors")
    if missing:
        raise FileNotFoundError(f"incomplete {method}: {', '.join(missing)}")
    item = metadata(path)
    expected_variant = "nbs_v19" if method == "nbs" else "adalora"
    checks = {
        "variant": expected_variant,
        "data_seed": data_seed,
        "effective_rank_budget": 1536,
        "physical_rank": 32,
    }
    mismatches = {
        key: (item.get(key), value)
        for key, value in checks.items() if item.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{method} checkpoint mismatch: {mismatches}")
    return item


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize_method(rows, method):
    group = sorted(
        (row for row in rows if row.get("method") == method),
        key=lambda row: int(row["evaluation_seed"]),
    )
    if [int(row["evaluation_seed"]) for row in group] != list(SEEDS):
        raise ValueError(f"{method} does not have evaluation seeds 1, 2, 3")
    result = {
        "method": method,
        "label": lora_eval.METHODS[method]["label"],
        "checkpoint_training_data_seed": int(group[0]["checkpoint_training_data_seed"]),
        "num_seeds": 3,
        "evaluation_seeds": "1,2,3",
        "rank_budget": 1536,
        "physical_rank": lora_eval.METHODS[method]["physical_rank"],
        "evaluation_rng_mode": "per-episode",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
    }
    for metric in lora_eval.SUMMARY_METRICS:
        values = [finite(row.get(metric)) for row in group]
        if all(value is not None for value in values):
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values)
    return result


def evaluate_lora(args, method, checkpoint, checkpoint_data_seed):
    output = args.output_dir / "fixed_lora" / "lora_methods_per_seed.csv"
    rows = lora_eval.load_rows(output)
    completed = {
        (row.get("method"), int(row.get("evaluation_seed", -1)))
        for row in rows
        if row.get("checkpoint_dir") == str(checkpoint.resolve())
    }
    command_args = argparse.Namespace(
        base_model_dir=args.base_model_dir,
        exp_pool_path=args.exp_pool_path,
        rank_budget=1536,
        trace=args.trace,
        trace_num=args.trace_num,
        video=args.video,
        device=args.device,
    )
    for seed in SEEDS:
        key = (method, seed)
        if key in completed:
            print(f"[{method} seed={seed}] already complete; skipping", flush=True)
            continue
        command = lora_eval.build_command(command_args, method, checkpoint, seed)
        print(f"[{method} seed={seed}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        started_at = time.time() - 1.0
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = lora_eval.newest_metrics(started_at)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "method": method,
            "label": lora_eval.METHODS[method]["label"],
            "checkpoint_training_data_seed": checkpoint_data_seed,
            "evaluation_seed": seed,
            "rank_budget": 1536,
            "physical_rank": lora_eval.METHODS[method]["physical_rank"],
            "checkpoint_dir": str(checkpoint.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            **lora_eval.scalar_metrics(metrics),
        })
        lora_eval.write_rows(output, rows)
    if args.dry_run:
        return
    summary_path = args.output_dir / "fixed_lora" / f"{method}_three_seed_summary.csv"
    lora_eval.write_rows(summary_path, [summarize_method(rows, method)])
    print(f"[{method}] summary: {summary_path.resolve()}", flush=True)


def nbs_module_args(args, checkpoint):
    return argparse.Namespace(
        checkpoint_dir=checkpoint,
        base_model_dir=args.base_model_dir,
        exp_pool_path=args.exp_pool_path,
        rank_budget=1536,
        physical_rank=32,
        rank_config=Path("configs/nbs_v19_rank_config.json"),
        trace=args.trace,
        trace_num=args.trace_num,
        video=args.video,
        device=args.device,
        evaluation_rng_mode="per-episode",
        run_tag="server2_data3_fp16_prescaled_qk",
        fp16_numeric_safeguards=True,
        fp16_selective_clamp=True,
        fp16_selective_clamp_threshold=60000.0,
        fp16_attention_fp32_scores=False,
        fp16_attention_prescaled_qk=True,
        nbs_compaction_rtol=0.05,
        nbs_compaction_atol=0.01,
        dry_run=args.dry_run,
    )


def evaluate_nbs_modules(args, checkpoint):
    output = args.output_dir / "nbs_data3_modules" / "per_seed_results.csv"
    rows = module_sweep.load_rows(output) if args.resume else []
    run_args = nbs_module_args(args, checkpoint)
    failures = []
    for seed in SEEDS:
        for spec in latest_modules.TARGET_SPECS:
            try:
                rows = module_sweep.run_specs(
                    run_args, (spec,), seed, output, rows,
                )
            except Exception as error:  # keep the unattended queue alive
                failures.append({
                    "seed": seed,
                    "experiment": spec["name"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                print(
                    f"[{seed}:{spec['name']}] FAILED: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr, flush=True,
                )
    if args.dry_run:
        return
    if failures:
        atomic_json(
            args.output_dir / "nbs_data3_modules" / "failed_runs.json",
            failures,
        )
    summary = latest_modules.summarize(rows)
    summary_path = args.output_dir / "nbs_data3_modules" / "three_seed_summary.csv"
    module_sweep.write_rows(summary_path, summary)
    print(f"[nbs modules] summary: {summary_path.resolve()}", flush=True)


def run_stage(args, state, state_path, name, action):
    previous = state["stages"].get(name, {})
    if previous.get("status") == "complete":
        print(f"[{name}] already complete; skipping", flush=True)
        return
    print(f"\n===== {name} =====", flush=True)
    started = time.time()
    try:
        action()
    except Exception as error:  # deliberately continue to independent stages
        state["stages"][name] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "started_at": started,
            "finished_at": time.time(),
        }
        print(
            f"[{name}] FAILED: {type(error).__name__}: {error}",
            file=sys.stderr, flush=True,
        )
    else:
        state["stages"][name] = {
            "status": "complete",
            "started_at": started,
            "finished_at": time.time(),
        }
    if not args.dry_run:
        atomic_json(state_path, state)


def checkpoint_from_state(state, key):
    value = state["checkpoints"].get(key)
    if not value:
        raise RuntimeError(f"checkpoint unavailable because stage failed: {key}")
    return Path(value)


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
    state_path, state = load_state(args)

    run_stage(
        args, state, state_path, "train_eva_data1",
        lambda: train_new_checkpoint(
            args, state, state_path, "eva_data1", EVA_DATA1,
        ),
    )
    run_stage(
        args, state, state_path, "evaluate_eva_data1",
        lambda: evaluate_lora(
            args, "eva", checkpoint_from_state(state, "eva_data1"), 1,
        ),
    )

    def validate_data3():
        for method in ("nbs", "adalora"):
            checkpoint = discover_existing(method, 3)
            validate_existing(checkpoint, method, 3)
            state["checkpoints"][f"{method}_data3"] = str(checkpoint.resolve())
            if not args.dry_run:
                atomic_json(state_path, state)
            print(f"[{method} data3] checkpoint: {checkpoint}", flush=True)

    run_stage(
        args, state, state_path, "validate_existing_data3", validate_data3,
    )
    run_stage(
        args, state, state_path, "evaluate_adalora_data3",
        lambda: evaluate_lora(
            args, "adalora",
            checkpoint_from_state(state, "adalora_data3"), 3,
        ),
    )
    run_stage(
        args, state, state_path, "evaluate_nbs_data3_modules",
        lambda: evaluate_nbs_modules(
            args, checkpoint_from_state(state, "nbs_data3"),
        ),
    )
    run_stage(
        args, state, state_path, "train_uniform_data3",
        lambda: train_new_checkpoint(
            args, state, state_path, "uniform_data3", UNIFORM_DATA3,
        ),
    )
    run_stage(
        args, state, state_path, "evaluate_uniform_data3",
        lambda: evaluate_lora(
            args, "uniform",
            checkpoint_from_state(state, "uniform_data3"), 3,
        ),
    )

    report = {
        name: state["stages"].get(name, {"status": "not-run"})
        for name in STAGE_ORDER
    }
    if not args.dry_run:
        atomic_json(args.output_dir / "final_stage_report.json", report)
    print("\n===== final stage report =====", flush=True)
    for name, item in report.items():
        print(f"{name}: {item['status']}", flush=True)
    print(f"OUTPUT={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
