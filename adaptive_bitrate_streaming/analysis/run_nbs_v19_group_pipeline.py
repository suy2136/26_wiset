"""Shared sequential train/test pipeline for ABR NBS and LoRA sweeps."""

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


def experiment_method(experiment):
    return experiment.get("method", "nbs")


def experiment_seed(experiment):
    return int(experiment.get("seed", 1))


def expected_variant(experiment):
    return {
        "nbs": "nbs_v19",
        "uniform_lora": "uniform_lora",
        "adalora": "adalora",
        "shapley": "shapley",
        "eva": "eva",
    }[experiment_method(experiment)]


def expected_checkpoint_role(experiment):
    # An early AdaLoRA best can precede the target-budget phase.  Its final
    # checkpoint is used for an exact capacity-matched comparison.
    return (
        "final"
        if experiment_method(experiment) in ("adalora", "shapley")
        else "best"
    )


def eva_state_dir(args, experiment):
    configured = experiment.get("eva_state_path")
    if configured:
        path = Path(configured)
        return path.parent if path.suffix else path
    return args.output.parent / f"{args.output.stem}_eva" / experiment["name"]


def build_eva_precompute_command(args, experiment):
    command = [
        sys.executable, "analysis/precompute_abr_eva.py", "--fp16",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--output-dir", str(eva_state_dir(args, experiment).resolve()),
        "--device", args.device,
        "--seed", str(experiment_seed(experiment)),
        "--rank-budget", str(experiment["rank_budget"]),
        "--min-rank", str(experiment.get("min_rank", 2)),
        "--max-rank", str(experiment.get("max_rank", 32)),
        "--metric", experiment.get("eva_metric", "ratio"),
        "--similarity-threshold",
        str(experiment.get("eva_similarity_threshold", 0.99)),
        "--min-batches", str(experiment.get("eva_min_batches", 2)),
        "--max-batches", str(experiment.get("eva_max_batches", 128)),
    ]
    if experiment.get("eva_allow_unconverged", False):
        command.append("--allow-unconverged")
    return command


def build_training_command(args, experiment):
    method = experiment_method(experiment)
    command = [
        sys.executable, "run_plm.py", "--adapt", "--fp16",
        "--seed", str(experiment_seed(experiment)),
        "--plm-type", "llama", "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--rank", str(experiment["physical_rank"]),
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
    if method == "nbs":
        command[3:3] = [
            "--nbs-v19",
            "--nbs-rank-budget", str(experiment["rank_budget"]),
            "--nbs-rank-config", experiment["rank_config"],
            "--nbs-rollback-backup-device", args.nbs_rollback_backup_device,
            "--nbs-max-rollback-backup-mib",
            str(args.nbs_max_rollback_backup_mib),
            "--nbs-update-ratio-warning", str(args.nbs_update_ratio_warning),
            "--nbs-max-update-ratio", str(args.nbs_max_update_ratio),
            "--nbs-update-ratio-floor", str(args.nbs_update_ratio_floor),
            "--nbs-max-update-rms", str(args.nbs_max_update_rms),
            "--nbs-rollback-lr-factor", str(args.nbs_rollback_lr_factor),
            "--nbs-max-consecutive-rollbacks",
            str(args.nbs_max_consecutive_rollbacks),
        ]
    elif method in ("adalora", "shapley"):
        command[3:3] = [
            "--lora-method", method,
            "--adalora-rank-budget", str(experiment["rank_budget"]),
            "--adalora-allocation-interval",
            str(experiment.get("allocation_interval", 10)),
            "--adalora-schedule-epochs",
            str(experiment.get(
                "adalora_schedule_epochs", args.early_stopping_min_epochs
            )),
        ]
        if method == "shapley":
            command[3:3] = [
                "--shapley-permutations",
                str(experiment.get("shapley_permutations", 1)),
                "--shapley-validation-batches",
                str(experiment.get("shapley_validation_batches", 1)),
                "--shapley-truncate-fraction",
                str(experiment.get("shapley_truncate_fraction", 0.05)),
            ]
            if not experiment.get("shapley_antithetic", True):
                command.insert(3, "--no-shapley-antithetic")
    elif method == "eva":
        command[3:3] = [
            "--lora-method", "eva",
            "--eva-state-path",
            str(eva_state_dir(args, experiment).resolve()),
        ]
    elif method != "uniform_lora":
        raise ValueError(f"unsupported experiment method: {method}")
    return command


def build_test_command(args, experiment, checkpoint):
    method = experiment_method(experiment)
    command = [
        sys.executable, "run_plm.py", "--test", "--fp16",
        "--seed", str(experiment_seed(experiment)),
        "--plm-type", "llama", "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--model-dir", str(checkpoint.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--rank", str(experiment["physical_rank"]),
        "--lr", str(experiment["lr"]),
        "--lr-schedule", "cosine",
        "--warmup-steps", str(experiment["warmup_steps"]),
        "--num-epochs", str(args.num_epochs),
        "--trace", args.test_trace, "--trace-num", str(args.trace_num),
        "--video", args.video, "--fixed-order",
        "--device", args.device, "--device-out", args.device,
        "--temporal-selector", "none", "--token-selector", "none",
        "--speculative-draft-steps", "0",
    ]
    if method == "nbs":
        command[3:3] = [
            "--nbs-v19",
            "--nbs-rank-budget", str(experiment["rank_budget"]),
            "--nbs-rank-config", experiment["rank_config"],
        ]
        command.append("--nbs-compact-inference")
    elif method in ("adalora", "shapley"):
        command[3:3] = [
            "--lora-method", method,
            "--adalora-rank-budget", str(experiment["rank_budget"]),
            "--adalora-allocation-interval",
            str(experiment.get("allocation_interval", 10)),
        ]
        if method == "shapley":
            command[3:3] = [
                "--shapley-permutations",
                str(experiment.get("shapley_permutations", 1)),
                "--shapley-validation-batches",
                str(experiment.get("shapley_validation_batches", 1)),
                "--shapley-truncate-fraction",
                str(experiment.get("shapley_truncate_fraction", 0.05)),
            ]
            if not experiment.get("shapley_antithetic", True):
                command.insert(3, "--no-shapley-antithetic")
    elif method == "eva":
        state_path = checkpoint / "eva_state.pt"
        command[3:3] = [
            "--lora-method", "eva",
            "--eva-state-path", str(state_path.resolve()),
        ]
    elif method != "uniform_lora":
        raise ValueError(f"unsupported experiment method: {method}")
    return command


def discover_best_checkpoint(experiment, started_at):
    lr_marker = f"_lr_{experiment['lr']}_"
    variant = expected_variant(experiment)
    candidates = []
    for metadata_path in MODEL_ROOT.rglob("checkpoint_metadata.json"):
        text = str(metadata_path)
        if (
            lr_marker not in text
            or metadata_path.stat().st_mtime < started_at
        ):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("role") == expected_checkpoint_role(experiment)
            and metadata.get("variant") == variant
            and metadata.get("seed") == experiment_seed(experiment)
            and metadata.get("physical_rank") == experiment["physical_rank"]
            and metadata.get("effective_rank_budget")
            == experiment["rank_budget"]
        ):
            candidates.append(metadata_path.parent)
    if not candidates:
        raise RuntimeError(
            f"{experiment['name']} produced no new matching "
            f"{expected_checkpoint_role(experiment)} checkpoint"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_checkpoint(path, experiment):
    method = experiment_method(experiment)
    required = [
        "adapter_config.json", "modules_except_plm.bin",
        "checkpoint_metadata.json",
    ]
    if method == "nbs":
        required.append("nash_rank_allocator.pt")
    if method == "eva":
        required.append("eva_state.pt")
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
    if metadata.get("variant") != expected_variant(experiment):
        raise ValueError(f"{experiment['name']} checkpoint method mismatch")
    if metadata.get("role") != expected_checkpoint_role(experiment):
        raise ValueError(f"{experiment['name']} checkpoint role mismatch")
    if metadata.get("seed") != experiment_seed(experiment):
        raise ValueError(f"{experiment['name']} checkpoint seed mismatch")
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
        "seeds": {
            item["name"]: experiment_seed(item) for item in experiments
        },
        "early_stopping": {
            "patience": args.early_stopping_patience,
            "min_epochs": args.early_stopping_min_epochs,
            "min_delta": args.early_stopping_min_delta,
        },
        "numeric_safety": {
            "rollback_backup_device": args.nbs_rollback_backup_device,
            "max_rollback_backup_mib": args.nbs_max_rollback_backup_mib,
            "update_ratio_warning": args.nbs_update_ratio_warning,
            "max_update_ratio": args.nbs_max_update_ratio,
            "update_ratio_floor": args.nbs_update_ratio_floor,
            "max_update_rms": args.nbs_max_update_rms,
            "rollback_lr_factor": args.nbs_rollback_lr_factor,
            "max_consecutive_rollbacks": args.nbs_max_consecutive_rollbacks,
        },
        "features": {
            "temporal_selector": "none", "token_selector": "none",
            "speculative_draft_steps": 0,
            "nbs_compact_inference": any(
                experiment_method(item) == "nbs" for item in experiments
            ),
        },
    }


def load_state(path, resume, run_signature):
    if not resume or not path.is_file():
        return {"signature": run_signature, "runs": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    # States written before transactional optimizer guards did not record the
    # safety configuration. Preserve completed runs and resume the first
    # unfinished experiment with the newly requested guards.
    if "numeric_safety" not in state.get("signature", {}):
        state["signature"]["numeric_safety"] = run_signature[
            "numeric_safety"
        ]
    # Migrate seed-1-only states produced before experiments could override
    # the common seed (for example the new C_SEED2 replication).
    if "seeds" not in state.get("signature", {}):
        legacy_seed = state["signature"].pop("seed", 1)
        state["signature"]["seeds"] = {
            item["name"]: legacy_seed for item in run_signature["experiments"]
        }
    # Calibration caps do not alter a completed allocator checkpoint.  Allow
    # an unfinished EVA run to adopt a larger convergence cap (and an
    # explicitly recorded unconverged-at-cap fallback) without discarding
    # already completed experiments in the same sequential server job.
    saved_experiments = state.get("signature", {}).get("experiments", [])
    requested_experiments = run_signature.get("experiments", [])
    if len(saved_experiments) == len(requested_experiments):
        calibration_fields = {
            "eva_max_batches", "eva_allow_unconverged",
        }
        for saved, requested in zip(saved_experiments, requested_experiments):
            if saved.get("name") != requested.get("name"):
                continue
            run = state.get("runs", {}).get(saved["name"], {})
            if (
                requested.get("method") == "eva"
                and not run.get("checkpoint_dir")
                and run.get("status") != "complete"
            ):
                for field in calibration_fields:
                    if field in requested:
                        saved[field] = requested[field]
                    else:
                        saved.pop(field, None)
    if state.get("signature") != run_signature:
        raise ValueError("saved group state does not match current experiment setup")
    state["signature"] = run_signature
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
            "method": experiment_method(experiment),
            "seed": experiment_seed(experiment),
            "checkpoint_dir": run["checkpoint_dir"],
            "checkpoint_metadata_path": run.get("checkpoint_metadata_path"),
            "allocator_artifacts": ";".join(
                run.get("allocator_artifacts", [])
            ),
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
    parser.add_argument(
        "--nbs-rollback-backup-device", choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--nbs-max-rollback-backup-mib", type=float, default=2048.0,
    )
    parser.add_argument("--nbs-update-ratio-warning", type=float, default=0.01)
    parser.add_argument("--nbs-max-update-ratio", type=float, default=0.05)
    parser.add_argument("--nbs-update-ratio-floor", type=float, default=0.01)
    parser.add_argument("--nbs-max-update-rms", type=float, default=0.01)
    parser.add_argument("--nbs-rollback-lr-factor", type=float, default=0.5)
    parser.add_argument(
        "--nbs-max-consecutive-rollbacks", type=int, default=3,
    )
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
        if experiment_method(experiment) == "eva":
            state_path = eva_state_dir(args, experiment) / "eva_state.pt"
            if not state_path.is_file():
                precompute_command = build_eva_precompute_command(
                    args, experiment
                )
                print(
                    f"[{name}:eva] {shlex.join(precompute_command)}",
                    flush=True,
                )
                if not args.dry_run:
                    subprocess.run(
                        precompute_command, cwd=ABR_ROOT, check=True
                    )
            elif not args.dry_run:
                print(
                    f"[{name}:eva] state already available: {state_path}",
                    flush=True,
                )
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
        if experiment_method(experiment) == "nbs":
            if not metrics.get("nbs_compact_inference"):
                raise RuntimeError(f"{name} inference did not use NBS compaction")
            if not metrics.get("nbs_compaction_logits_equivalent"):
                raise RuntimeError(f"{name} compact logits equivalence failed")
        elif metrics.get("nbs_compact_inference"):
            raise RuntimeError(f"{name} unexpectedly used NBS compaction")

        metrics_dir.mkdir(parents=True, exist_ok=True)
        saved_metrics = metrics_dir / f"{name}_selector_metrics.json"
        shutil.copy2(source_metrics, saved_metrics)
        saved_metadata = metrics_dir / f"{name}_checkpoint_metadata.json"
        shutil.copy2(checkpoint / "checkpoint_metadata.json", saved_metadata)
        auxiliary_artifacts = []
        if experiment_method(experiment) in ("adalora", "shapley"):
            diagnostic_name = (
                f"{experiment_method(experiment)}_rank_diagnostics.jsonl"
            )
            diagnostic_source = checkpoint.parent / diagnostic_name
            if diagnostic_source.is_file():
                diagnostic_destination = metrics_dir / (
                    f"{name}_{diagnostic_name}"
                )
                shutil.copy2(
                    diagnostic_source, diagnostic_destination
                )
                auxiliary_artifacts.append(
                    str(diagnostic_destination.resolve())
                )
        if experiment_method(experiment) == "eva":
            source_dir = eva_state_dir(args, experiment)
            for artifact_name in (
                "rank_pattern.json", "explained_variance.csv", "metadata.json"
            ):
                source = source_dir / artifact_name
                if source.is_file():
                    destination = metrics_dir / f"{name}_{artifact_name}"
                    shutil.copy2(source, destination)
                    auxiliary_artifacts.append(str(destination.resolve()))
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
            "checkpoint_metadata_path": str(saved_metadata.resolve()),
            "allocator_artifacts": auxiliary_artifacts,
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
            f"[{name}] test QoE={metrics['mean_reward']:.6f} "
            f"latency={metrics['inference_latency_mean_ms']:.3f} ms",
            flush=True,
        )

    if not args.dry_run:
        rows = result_rows(state, experiments)
        write_results(args.output, rows, run_signature)
        print(f"Results saved at: {args.output.resolve()}")
