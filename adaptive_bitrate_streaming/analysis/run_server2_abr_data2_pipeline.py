"""Train and evaluate the complete ABR data-seed-2 experiment queue.

The queue is additive and restartable. Existing checkpoints are never edited;
new training and evaluation artifacts use run-specific tags and output paths.
Rank-budget mismatches are reported and recorded, but do not block evaluation.
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
MODEL_ROOT = ABR_ROOT / "data/ft_plms"
DEFAULT_OUTPUT = ABR_ROOT / "artifacts/results/server2_abr_data2_pipeline"
SEEDS = (1, 2, 3)
TARGET_BUDGET = 1536
TRAINING_DATA_SEED = 2
METHOD_ORDER = ("nbs", "uniform", "adalora", "shapley", "eva")
METHOD_LABELS = {
    "nbs": "NBS-LoRA compact", "uniform": "Uniform LoRA r24",
    "adalora": "AdaLoRA", "shapley": "ShapLoRA", "eva": "EVA",
}
STAGE_ORDER = (
    "train_nbs", "evaluate_nbs_modules",
    "train_uniform", "evaluate_uniform",
    "train_adalora", "evaluate_adalora",
    "train_shapley", "evaluate_shapley",
    "train_eva", "evaluate_eva",
    "write_lora_summary",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model-dir", type=Path, default=training.DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path, default=training.DEFAULT_EXP_POOL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def experiment_for(args, method: str) -> dict:
    common = {
        "name": f"SERVER2_ABR_DATA2_{method.upper()}",
        "method": "uniform_lora" if method == "uniform" else method,
        "rank_budget": TARGET_BUDGET,
        "physical_rank": 24 if method == "uniform" else 32,
        "lr": 2e-4,
        "warmup_steps": 500,
        "seed": 1,
        "lora_seed": 1,
        "data_seed": TRAINING_DATA_SEED,
        "run_tag": f"server2_abr_data2_{method}_{safe_tag(args.output_dir.name)}",
    }
    if method == "nbs":
        common.update({
            "rank_config": "configs/nbs_v19_rank_config.json",
            "nbs_allocation_audit": True,
        })
    elif method in {"adalora", "shapley"}:
        common.update({"allocation_interval": 10, "adalora_schedule_epochs": 20})
        if method == "shapley":
            common.update({
                "shapley_permutations": 1,
                "shapley_validation_batches": 1,
                "shapley_truncate_fraction": 0.05,
                "shapley_antithetic": True,
            })
    elif method == "eva":
        common.update({
            "min_rank": 2, "max_rank": 32,
            "eva_metric": "ratio", "eva_similarity_threshold": 0.99,
            "eva_min_batches": 2, "eva_max_batches": 512,
            "eva_allow_unconverged": True,
        })
    return common


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(args) -> dict:
    return {
        "pipeline": "server2_abr_data2_v1",
        "stage_order": list(STAGE_ORDER),
        "base_model_dir": str(args.base_model_dir.resolve()),
        "exp_pool_path": str(args.exp_pool_path.resolve()),
        "device": args.device,
        "trace": args.trace,
        "trace_num": args.trace_num,
        "video": args.video,
        "training_seed": 1,
        "lora_seed": 1,
        "training_data_seed": TRAINING_DATA_SEED,
        "evaluation_seeds_and_data_seeds": list(SEEDS),
        "evaluation_rng_mode": "per-episode",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "target_rank_budget": TARGET_BUDGET,
        "experiments": [experiment_for(args, method) for method in METHOD_ORDER],
        "modules": [spec["name"] for spec in latest_modules.TARGET_SPECS],
    }


def load_state(args):
    path = args.output_dir / "pipeline_state.json"
    expected = signature(args)
    if path.is_file():
        if not args.resume:
            raise FileExistsError(f"output exists: {args.output_dir}; use --resume")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("signature") != expected:
            raise ValueError("resume state differs from the current configuration")
        return path, state
    state = {"signature": expected, "stages": {}, "checkpoints": {}}
    if not args.dry_run:
        atomic_json(path, state)
    return path, state


def active_rank(value) -> int:
    if isinstance(value, list):
        return sum(bool(item) for item in value)
    if isinstance(value, bool):
        return int(value)
    return int(value)


def inspect_budget(path: Path, method: str, experiment: dict) -> dict:
    required = ["adapter_config.json", "modules_except_plm.bin", "checkpoint_metadata.json"]
    if method == "nbs":
        required.append("nash_rank_allocator.pt")
    if method == "eva":
        required.append("eva_state.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if not any((path / name).is_file() for name in (
        "adapter_model.bin", "adapter_model.safetensors",
    )):
        missing.append("adapter weights")
    if missing:
        raise FileNotFoundError(f"incomplete {method} checkpoint: {missing}")

    metadata = json.loads((path / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    structural_checks = {
        "variant": training.expected_variant(experiment),
        "role": training.expected_checkpoint_role(experiment),
        "seed": 1,
        "lora_seed": 1,
        "data_seed": TRAINING_DATA_SEED,
        "physical_rank": experiment["physical_rank"],
        "run_tag": experiment["run_tag"],
    }
    mismatches = {
        key: (metadata.get(key, metadata.get("seed") if key in {"lora_seed", "data_seed"} else None), expected)
        for key, expected in structural_checks.items()
        if metadata.get(key, metadata.get("seed") if key in {"lora_seed", "data_seed"} else None) != expected
    }
    if mismatches:
        raise ValueError(f"{method} checkpoint identity mismatch: {mismatches}")

    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    rank_pattern = config.get("rank_pattern") or {}
    active_ranks = metadata.get("active_ranks") or {}
    if rank_pattern:
        actual = sum(active_rank(value) for value in rank_pattern.values())
    elif method == "uniform":
        actual = int(config.get("r", experiment["physical_rank"])) * 64
    elif isinstance(active_ranks, dict) and active_ranks:
        actual = sum(int(value) for value in active_ranks.values())
    else:
        actual = int(metadata.get("effective_rank_budget", TARGET_BUDGET))
    matched = actual == TARGET_BUDGET
    if not matched:
        print(
            f"[{method}] WARNING: active rank is {actual}, target is {TARGET_BUDGET}; "
            f"evaluation will continue and record the mismatch: {path}",
            file=sys.stderr, flush=True,
        )
    return {
        "checkpoint": str(path.resolve()),
        "active_rank_total": actual,
        "target_rank_budget": TARGET_BUDGET,
        "budget_match": matched,
        "budget_note": "matched_1536" if matched else f"nonmatching_{actual}_target_1536",
        "metadata_effective_rank_budget": metadata.get("effective_rank_budget"),
    }


def training_args(args, key: str):
    argv = [
        "--base-model-dir", str(args.base_model_dir),
        "--exp-pool-path", str(args.exp_pool_path),
        "--device", args.device,
        "--state-file", str(args.output_dir / f"training/{key}_state.json"),
        "--output", str(args.output_dir / f"training/{key}_result.csv"),
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
        argv,
        state_file=args.output_dir / f"training/{key}_state.json",
        output_file=args.output_dir / f"training/{key}_result.csv",
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    return parsed


def add_prescaled_qk(command: list[str]) -> list[str]:
    command = [item for item in command if item != "--fp16-attention-fp32-scores"]
    if "--fp16-attention-prescaled-qk" not in command:
        command.append("--fp16-attention-prescaled-qk")
    return command


def discover_checkpoint(experiment: dict, started_at: float = 0.0) -> Path | None:
    candidates = []
    for metadata_path in MODEL_ROOT.rglob("checkpoint_metadata.json"):
        if metadata_path.stat().st_mtime < started_at:
            continue
        try:
            item = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            item.get("variant") == training.expected_variant(experiment)
            and item.get("role") == training.expected_checkpoint_role(experiment)
            and int(item.get("seed", -1)) == 1
            and int(item.get("lora_seed", item.get("seed", -1))) == 1
            and int(item.get("data_seed", item.get("seed", -1))) == TRAINING_DATA_SEED
            and int(item.get("physical_rank", -1)) == int(experiment["physical_rank"])
            and item.get("run_tag") == experiment["run_tag"]
        ):
            candidates.append(metadata_path.parent)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def train_method(args, state, state_path: Path, method: str) -> Path:
    key = f"{method}_data2"
    experiment = experiment_for(args, method)
    saved = state["checkpoints"].get(key)
    if saved:
        checkpoint = Path(saved)
        inspect_budget(checkpoint, method, experiment)
        return checkpoint
    if args.resume:
        recovered = discover_checkpoint(experiment)
        if recovered is not None:
            inspection = inspect_budget(recovered, method, experiment)
            state["checkpoints"][key] = str(recovered.resolve())
            state.setdefault("checkpoint_inspections", {})[key] = inspection
            atomic_json(state_path, state)
            print(f"[{method}] recovered completed checkpoint: {recovered}", flush=True)
            return recovered

    train_args = training_args(args, key)
    if method == "eva":
        eva_state = training.eva_state_dir(train_args, experiment) / "eva_state.pt"
        if not eva_state.is_file():
            command = training.build_eva_precompute_command(train_args, experiment)
            print(f"[{method}:eva] {shlex.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ABR_ROOT, check=True)

    command = add_prescaled_qk(training.build_training_command(train_args, experiment))
    print(f"[{method}:train] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return Path(f"/dry-run/{key}")
    started_at = time.time() - 1.0
    subprocess.run(command, cwd=ABR_ROOT, check=True)
    checkpoint = discover_checkpoint(experiment, started_at)
    if checkpoint is None:
        raise FileNotFoundError(f"new final checkpoint not found for {method}")
    inspection = inspect_budget(checkpoint, method, experiment)
    state["checkpoints"][key] = str(checkpoint.resolve())
    state.setdefault("checkpoint_inspections", {})[key] = inspection
    atomic_json(state_path, state)
    print(f"[{method}] checkpoint: {checkpoint}", flush=True)
    return checkpoint


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def set_option(command: list[str], option: str, value: str) -> None:
    if option in command:
        command[command.index(option) + 1] = value
    else:
        command.extend([option, value])


def evaluate_lora(args, method: str, checkpoint: Path) -> dict:
    experiment = experiment_for(args, method)
    inspection = inspect_budget(checkpoint, method, experiment)
    output = args.output_dir / "lora_methods/per_seed_results.csv"
    rows = lora_eval.load_rows(output) if args.resume else []
    completed = {
        (row.get("method"), int(row.get("evaluation_seed", -1)))
        for row in rows
        if row.get("checkpoint_dir") == str(checkpoint.resolve())
        and row.get("status") == "complete"
    }
    command_args = argparse.Namespace(
        base_model_dir=args.base_model_dir,
        exp_pool_path=args.exp_pool_path,
        rank_budget=TARGET_BUDGET,
        trace=args.trace,
        trace_num=args.trace_num,
        video=args.video,
        device=args.device,
    )
    for seed in SEEDS:
        if (method, seed) in completed:
            print(f"[{method} seed={seed}] already complete; skipping", flush=True)
            continue
        command = add_prescaled_qk(lora_eval.build_command(command_args, method, checkpoint, seed))
        set_option(command, "--run-tag", f"server2_abr_data2_lora_{safe_tag(args.output_dir.name)}")
        print(f"[{method} seed={seed}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        row = {
            "method": method, "label": METHOD_LABELS[method],
            "checkpoint_training_data_seed": TRAINING_DATA_SEED,
            "evaluation_seed": seed,
            "rank_budget": TARGET_BUDGET,
            "physical_rank": experiment["physical_rank"],
            "checkpoint_dir": str(checkpoint.resolve()),
            "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
            "evaluation_rng_mode": "per-episode",
            "status": "failed",
            **inspection,
        }
        try:
            started_at = time.time() - 1.0
            subprocess.run(command, cwd=ABR_ROOT, check=True)
            metrics_path = lora_eval.newest_metrics(started_at)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row.update(lora_eval.scalar_metrics(metrics))
            row["metrics_path"] = str(metrics_path.resolve())
            row["status"] = "complete"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        lora_eval.write_rows(output, rows)
        if row["status"] != "complete":
            raise RuntimeError(row["error"])
    if args.dry_run:
        return {}
    group = sorted(
        (row for row in rows if row.get("method") == method and row.get("status") == "complete"),
        key=lambda row: int(row["evaluation_seed"]),
    )
    if [int(row["evaluation_seed"]) for row in group] != list(SEEDS):
        raise RuntimeError(f"incomplete three-seed result: {method}")
    summary = {
        "method": method, "label": METHOD_LABELS[method],
        "checkpoint_training_data_seed": TRAINING_DATA_SEED,
        "num_seeds": 3, "evaluation_seeds": "1,2,3",
        "evaluation_rng_mode": "per-episode",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        **inspection,
    }
    for metric in lora_eval.SUMMARY_METRICS:
        values = [finite(row.get(metric)) for row in group]
        if all(value is not None for value in values):
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
    path = args.output_dir / f"lora_methods/{method}_three_seed_summary.csv"
    lora_eval.write_rows(path, [summary])
    return summary


def nbs_module_args(args, checkpoint: Path):
    return argparse.Namespace(
        checkpoint_dir=checkpoint,
        base_model_dir=args.base_model_dir,
        exp_pool_path=args.exp_pool_path,
        rank_budget=TARGET_BUDGET,
        physical_rank=32,
        rank_config=Path("configs/nbs_v19_rank_config.json"),
        trace=args.trace,
        trace_num=args.trace_num,
        video=args.video,
        device=args.device,
        evaluation_rng_mode="per-episode",
        run_tag=f"server2_abr_data2_modules_{safe_tag(args.output_dir.name)}",
        fp16_numeric_safeguards=True,
        fp16_selective_clamp=True,
        fp16_selective_clamp_threshold=60000.0,
        fp16_attention_fp32_scores=False,
        fp16_attention_prescaled_qk=True,
        nbs_compaction_rtol=0.05,
        nbs_compaction_atol=0.01,
        dry_run=args.dry_run,
    )


def evaluate_nbs_modules(args, checkpoint: Path) -> list[dict]:
    experiment = experiment_for(args, "nbs")
    inspection = inspect_budget(checkpoint, "nbs", experiment)
    output = args.output_dir / "nbs_modules/per_seed_results.csv"
    rows = module_sweep.load_rows(output) if args.resume else []
    run_args = nbs_module_args(args, checkpoint)
    failures = []
    for seed in SEEDS:
        for spec in latest_modules.TARGET_SPECS:
            try:
                rows = module_sweep.run_specs(run_args, (spec,), seed, output, rows)
            except Exception as error:
                failures.append({
                    "evaluation_seed": seed, "experiment": spec["name"],
                    "error_type": type(error).__name__, "error": str(error),
                })
                print(
                    f"[{seed}:{spec['name']}] FAILED: {type(error).__name__}: {error}",
                    file=sys.stderr, flush=True,
                )
    if args.dry_run:
        return []
    if failures:
        atomic_json(args.output_dir / "nbs_modules/failed_runs.json", failures)
    summaries = latest_modules.summarize(rows)
    for summary in summaries:
        summary.update({
            "checkpoint_training_data_seed": TRAINING_DATA_SEED,
            "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
            **inspection,
        })
    path = args.output_dir / "nbs_modules/three_seed_summary.csv"
    module_sweep.write_rows(path, summaries)
    pure_rows = [row for row in rows if row.get("experiment") == "nbs_compact_only"]
    for row in pure_rows:
        if not bool(row.get("nbs_compaction_logits_equivalent")):
            raise RuntimeError(
                f"NBS compact equivalence failed for seed {row.get('data_seed')}"
            )
    print(f"[nbs modules] summary: {path.resolve()}", flush=True)
    return summaries


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_combined_lora_summary(args) -> None:
    module_rows = load_csv(args.output_dir / "nbs_modules/three_seed_summary.csv")
    baseline = next(
        (row for row in module_rows if row.get("experiment") == "nbs_compact_only"),
        None,
    )
    if baseline is None:
        raise FileNotFoundError("NBS compact three-seed baseline is unavailable")
    baseline = dict(baseline)
    baseline.update({"method": "nbs", "label": METHOD_LABELS["nbs"]})
    summaries = [baseline]
    for method in ("uniform", "adalora", "shapley", "eva"):
        rows = load_csv(args.output_dir / f"lora_methods/{method}_three_seed_summary.csv")
        if not rows:
            raise FileNotFoundError(f"three-seed summary unavailable: {method}")
        summaries.append(rows[0])
    lora_eval.write_rows(
        args.output_dir / "lora_methods/five_method_three_seed_summary.csv",
        summaries,
    )


def run_stage(args, state, state_path: Path, name: str, action) -> None:
    if state["stages"].get(name, {}).get("status") == "complete":
        print(f"[{name}] already complete; skipping", flush=True)
        return
    print(f"\n===== {name} =====", flush=True)
    started = time.time()
    if not args.dry_run:
        state["stages"][name] = {"status": "running", "started_at": started}
        atomic_json(state_path, state)
    try:
        action()
    except Exception as error:
        state["stages"][name] = {
            "status": "failed", "error_type": type(error).__name__,
            "error": str(error), "started_at": started, "finished_at": time.time(),
        }
        print(f"[{name}] FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    else:
        state["stages"][name] = {
            "status": "complete", "started_at": started, "finished_at": time.time(),
        }
    if not args.dry_run:
        atomic_json(state_path, state)


def checkpoint_from_state(state, method: str) -> Path:
    value = state["checkpoints"].get(f"{method}_data2")
    if not value:
        raise RuntimeError(f"checkpoint unavailable: {method}_data2")
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

    if args.dry_run:
        for method in METHOD_ORDER:
            experiment = experiment_for(args, method)
            command = add_prescaled_qk(
                training.build_training_command(training_args(args, f"{method}_data2"), experiment)
            )
            print(shlex.join(command))
        return

    run_stage(
        args, state, state_path, "train_nbs",
        lambda: train_method(args, state, state_path, "nbs"),
    )
    run_stage(
        args, state, state_path, "evaluate_nbs_modules",
        lambda: evaluate_nbs_modules(args, checkpoint_from_state(state, "nbs")),
    )
    for method in ("uniform", "adalora", "shapley", "eva"):
        run_stage(
            args, state, state_path, f"train_{method}",
            lambda method=method: train_method(args, state, state_path, method),
        )
        run_stage(
            args, state, state_path, f"evaluate_{method}",
            lambda method=method: evaluate_lora(
                args, method, checkpoint_from_state(state, method),
            ),
        )
    run_stage(
        args, state, state_path, "write_lora_summary",
        lambda: write_combined_lora_summary(args),
    )

    failed = [name for name, item in state["stages"].items() if item.get("status") == "failed"]
    report = {name: state["stages"].get(name, {"status": "not-run"}) for name in STAGE_ORDER}
    atomic_json(args.output_dir / "final_stage_report.json", report)
    print("\n===== final stage report =====", flush=True)
    for name, item in report.items():
        print(f"{name}: {item['status']}", flush=True)
    print(f"Failed stages: {failed or 'none'}", flush=True)
    print(f"OUTPUT={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
