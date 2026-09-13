"""Tune the VP projector, select Token-K on validation, then test 3 seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.evaluate_cached_patch_followups_compact import selector_command
from analysis.evaluate_cached_patch_selectors_compact import (
    absolute, common_command, run_case, run_streaming_command, write_rows,
)
from analysis.evaluate_vp_fixed_spec_patch_token_sweep import full_stack_command
from analysis.evaluate_vp_inference_module_pipeline import (
    set_option, speculative_command, token_command, trace_metrics, write_json,
)


TRAINING_CANDIDATES = (
    ("projector_2ep_lr5e-5", 2, 5e-5),
    ("projector_2ep_lr2e-5", 2, 2e-5),
    ("projector_3ep_lr2e-5", 3, 2e-5),
)
# VP history has exactly 10 positions.  Larger K values are invalid rather
# than "keep everything" and are therefore excluded before launching a run.
TOKEN_K_VALUES = (2, 4, 6, 8, 10)
VP_HISTORY_LENGTH = 10
SEEDS = (1, 2, 3)
PATCH_CONFIG = {
    "policy": "gated-k1", "threshold": 6.0,
    "max_skip": 0, "projector_cache": True,
}
FINAL_CASES = (
    ("pure_nbs_compact", "NBS compact only", "baseline"),
    ("best_projector_patch", "+ best-projector Patch", "patch"),
    ("best_token", "+ selected Token K", "token"),
    ("spec_g6_t0p4", "+ Spec G=6/T=0.4", "speculative"),
    ("best_full_stack", "+ best Patch + Token + Spec", "full_stack"),
)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--nbs-checkpoint", type=Path, required=True)
    result.add_argument("--compact-checkpoint", type=Path, required=True)
    result.add_argument("--initial-projector", type=Path, required=True)
    result.add_argument("--existing-one-epoch-projector", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--rank-budget", type=int, default=512)
    result.add_argument("--physical-rank", type=int, default=32)
    result.add_argument("--validation-samples", type=int, default=128)
    result.add_argument("--token-mae-tolerance-deg", type=float, default=0.1)
    result.add_argument("--latency-warmup-steps", type=int, default=5)
    result.add_argument("--projector-cache-max-entries", type=int, default=512)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def projector_file(path):
    if path.is_dir() and (path / "best_multimodal_projector.pth").is_file():
        return path / "best_multimodal_projector.pth"
    if path.is_dir() and (path / "modules_except_plm.bin").is_file():
        return path
    return path


def validate_inputs(args):
    for name in ("adapter_model.bin", "modules_except_plm.bin", "nash_rank_allocator.pt"):
        if not (args.nbs_checkpoint / name).is_file():
            raise FileNotFoundError(args.nbs_checkpoint / name)
    for name in ("compact_adapter.pt", "modules_except_plm.bin"):
        if not (args.compact_checkpoint / name).is_file():
            raise FileNotFoundError(args.compact_checkpoint / name)
    for path in (args.initial_projector, args.existing_one_epoch_projector):
        if not projector_file(path).exists():
            raise FileNotFoundError(f"projector absent: {path}")
    history = args.existing_one_epoch_projector / "projector_training_history.csv"
    if not history.is_file():
        raise FileNotFoundError(f"one-epoch history absent: {history}")
    required_videos = (1, 5, 9, 2, 6, 11, 15, 16, 13, 17, 21, 22, 26,
                       19, 23, 3, 7, 12, 10, 20, 27, 4, 8, 14, 18, 24, 25)
    missing = [v for v in required_videos
               if not (args.cache_dir / f"video{v}_patch_features.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"patch caches missing for videos: {missing}")


def training_command(args, output, epochs, lr):
    return [
        sys.executable, "analysis/train_validate_cached_patch_projector.py",
        "--nbs-checkpoint", str(args.nbs_checkpoint),
        "--projector-checkpoint", str(args.initial_projector),
        "--cache-dir", str(args.cache_dir), "--output-dir", str(output),
        "--device", args.device, "--rank-budget", str(args.rank_budget),
        "--physical-rank", str(args.physical_rank), "--epochs", str(epochs),
        "--lr", str(lr), "--policy", "k1",
        "--validation-policy", "gated-k1",
        "--validation-samples", str(args.validation_samples),
        "--cache-device", "cpu",
    ]


def read_history(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def train_candidates(args):
    root = args.output_dir / "projector_candidates"
    records = []
    for name, epochs, lr in TRAINING_CANDIDATES:
        output = root / name
        complete = all((output / item).is_file() for item in (
            "best_multimodal_projector.pth", "latest_multimodal_projector.pth",
            "projector_training_history.csv", "projector_training_manifest.json",
        ))
        command = training_command(args, output, epochs, lr)
        if args.dry_run:
            print(json.dumps({"stage": "projector_train", "name": name,
                              "command": command}), flush=True)
            continue
        if not (args.resume and complete):
            output.mkdir(parents=True, exist_ok=True)
            run_streaming_command(command, REPO_ROOT, output / "pipeline.log")
        history = read_history(output / "projector_training_history.csv")
        best = min(history, key=lambda row: float(row["valid_mae_deg"]))
        records.append({
            "candidate": name, "projector_path": str(output / "best_multimodal_projector.pth"),
            "epochs": epochs, "lr": lr, "best_epoch": int(best["epoch"]),
            "valid_mae_deg": float(best["valid_mae_deg"]),
            "valid_mse": float(best["valid_mse"]),
        })

    if args.dry_run:
        return None, records
    reference_history = read_history(
        args.existing_one_epoch_projector / "projector_training_history.csv"
    )
    existing_best = min(
        reference_history, key=lambda row: float(row["valid_mae_deg"])
    )
    records.extend([
        {
            "candidate": "original_projector",
            "projector_path": str(args.initial_projector), "epochs": 0, "lr": 0,
            "best_epoch": 0,
            "valid_mae_deg": float(reference_history[0]["initial_valid_mae_deg"]),
            "valid_mse": float(reference_history[0]["initial_valid_mse"]),
        },
        {
            "candidate": "existing_1ep_lr5e-5",
            "projector_path": str(projector_file(args.existing_one_epoch_projector)),
            "epochs": 1, "lr": 5e-5,
            "best_epoch": int(existing_best["epoch"]),
            "valid_mae_deg": float(existing_best["valid_mae_deg"]),
            "valid_mse": float(existing_best["valid_mse"]),
        },
    ])
    records.sort(key=lambda row: row["valid_mae_deg"])
    write_rows(args.output_dir / "projector_candidates.csv", records)
    selected = Path(records[0]["projector_path"])
    print(
        f"Selected projector: {records[0]['candidate']} "
        f"valid_MAE={records[0]['valid_mae_deg']:.6f}", flush=True,
    )
    return selected, records


def configure_common(args):
    args.projector_checkpoint = args.initial_projector
    return args


def apply_evaluation(command, split, seed):
    set_option(command, "--evaluation-split", split)
    set_option(command, "--seed", seed)
    set_option(command, "--lora-seed", 1)
    set_option(command, "--data-seed", seed)
    set_option(command, "--evaluation-rng-mode", "continuous")
    return command


def full_command(args, projector, result_dir, token_k, split, seed):
    original = args.projector_checkpoint
    args.projector_checkpoint = projector
    try:
        command = full_stack_command(
            args, args.compact_checkpoint, result_dir, PATCH_CONFIG, token_k
        )
    finally:
        args.projector_checkpoint = original
    return apply_evaluation(command, split, seed)


def sweep_token_k(args, projector):
    invalid = [value for value in TOKEN_K_VALUES if value > VP_HISTORY_LENGTH]
    if invalid:
        raise ValueError(
            f"Token K exceeds VP history length {VP_HISTORY_LENGTH}: {invalid}"
        )
    rows = []
    root = args.output_dir / "token_k_validation"
    for token_k in TOKEN_K_VALUES:
        result_dir = root / f"k{token_k}"
        command = full_command(args, projector, result_dir, token_k, "valid", 1)
        if args.dry_run:
            print(json.dumps({"stage": "token_k_validation", "k": token_k,
                              "command": command}), flush=True)
            continue
        row = {"token_k": token_k, "status": "failed"}
        try:
            row.update(run_case(command, result_dir, args.resume))
            row.update(trace_metrics(result_dir))
            row["status"] = "complete"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        write_rows(args.output_dir / "token_k_validation.csv", rows)
        if row["status"] != "complete":
            raise RuntimeError(row["error"])
    if args.dry_run:
        return None, rows
    best_mae = min(float(row["mae"]) for row in rows)
    eligible = [row for row in rows
                if float(row["mae"]) <= best_mae + args.token_mae_tolerance_deg]
    selected = min(eligible, key=lambda row: float(row["latency_mean_ms"]))
    write_json(args.output_dir / "token_k_selection.json", {
        "criterion": "lowest latency within best validation MAE + tolerance",
        "mae_tolerance_deg": args.token_mae_tolerance_deg,
        "best_validation_mae": best_mae,
        "selected": selected,
    })
    print(
        f"Selected Token K={selected['token_k']} validation MAE={float(selected['mae']):.6f} "
        f"latency={float(selected['latency_mean_ms']):.3f} ms", flush=True,
    )
    return int(selected["token_k"]), rows


def final_command(args, kind, projector, token_k, result_dir, seed):
    original = args.projector_checkpoint
    args.projector_checkpoint = projector
    try:
        if kind == "baseline":
            command = common_command(args, args.compact_checkpoint, result_dir)
        elif kind == "patch":
            command = selector_command(
                args, args.compact_checkpoint, result_dir, PATCH_CONFIG
            )
        elif kind == "token":
            command = token_command(
                args, args.compact_checkpoint, result_dir, token_k
            )
        elif kind == "speculative":
            command = speculative_command(
                args, args.compact_checkpoint, result_dir, 6, 0.4
            )
        elif kind == "full_stack":
            command = full_stack_command(
                args, args.compact_checkpoint, result_dir, PATCH_CONFIG, token_k
            )
        else:
            raise ValueError(kind)
    finally:
        args.projector_checkpoint = original
    return apply_evaluation(command, "test", seed)


def run_final_tests(args, projector, token_k):
    root = args.output_dir / "final_test"
    runs_path = args.output_dir / "final_test_per_seed.csv"
    rows = []
    if args.resume and runs_path.is_file():
        with runs_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    completed = {(row["case"], int(row["evaluation_seed"]))
                 for row in rows if row.get("status") == "complete"}
    for seed in SEEDS:
        for case, label, kind in FINAL_CASES:
            if (case, seed) in completed:
                continue
            result_dir = root / f"seed_{seed}" / case
            command = final_command(
                args, kind, projector, token_k, result_dir, seed
            )
            if args.dry_run:
                print(json.dumps({"stage": "final_test", "case": case,
                                  "seed": seed, "command": command}), flush=True)
                continue
            row = {
                "case": case, "label": label, "evaluation_seed": seed,
                "selected_token_k": token_k, "projector": str(projector),
                "status": "failed",
            }
            try:
                row.update(run_case(command, result_dir, args.resume))
                row.update(trace_metrics(result_dir))
                row["status"] = "complete"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            write_rows(runs_path, rows)
            print(f"[{case} seed={seed}] {row['status']} MAE={row.get('mae')} "
                  f"latency={row.get('latency_mean_ms')}", flush=True)
            if row["status"] != "complete":
                raise RuntimeError(row["error"])
    return rows


def summarize_final(rows):
    summaries = []
    for case, label, _ in FINAL_CASES:
        group = [row for row in rows if row["case"] == case
                 and row.get("status") == "complete"]
        if {int(row["evaluation_seed"]) for row in group} != set(SEEDS):
            raise RuntimeError(f"incomplete final test: {case}")
        summary = {"case": case, "label": label, "seed_count": 3}
        for metric in ("mae", "rmse", "latency_mean_ms",
                       "mean_selected_token_count", "mean_target_forward_count",
                       "mean_token_reduction_percent", "draft_acceptance_rate",
                       "selected_patches_mean", "visual_tokens_per_call"):
            values = [float(row[metric]) for row in group if row.get(metric) != ""]
            if len(values) == 3:
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(summary)
    baseline = summaries[0]
    for row in summaries:
        row["mae_change_percent_vs_nbs"] = (
            row["mae_mean"] / baseline["mae_mean"] - 1.0
        ) * 100.0
        row["latency_reduction_percent_vs_nbs"] = (
            1.0 - row["latency_mean_ms_mean"]
            / baseline["latency_mean_ms_mean"]
        ) * 100.0
    return summaries


def write_manifest(args):
    signature = {
        "nbs_checkpoint": str(args.nbs_checkpoint),
        "compact_checkpoint": str(args.compact_checkpoint),
        "initial_projector": str(args.initial_projector),
        "existing_one_epoch_projector": str(args.existing_one_epoch_projector),
        "cache_dir": str(args.cache_dir), "training_candidates": TRAINING_CANDIDATES,
        "token_k_values": TOKEN_K_VALUES, "patch": PATCH_CONFIG,
        "spec": {"gamma": 6, "threshold": 0.4},
        "selection_split": "valid", "final_split": "test", "seeds": SEEDS,
    }
    path = args.output_dir / "pipeline_manifest.json"
    if path.is_file() and args.resume:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != json.loads(json.dumps(signature)):
            raise ValueError("resume manifest differs from current configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    write_json(path, {"signature": signature})


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("nbs_checkpoint", "compact_checkpoint", "initial_projector",
                 "existing_one_epoch_projector", "cache_dir", "output_dir"):
        setattr(args, name, absolute(getattr(args, name)))
    configure_common(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        validate_inputs(args)
        write_manifest(args)
    selected_projector, _ = train_candidates(args)
    if args.dry_run:
        selected_projector = args.initial_projector
    selected_k, _ = sweep_token_k(args, selected_projector)
    if args.dry_run:
        selected_k = 8
    rows = run_final_tests(args, selected_projector, selected_k)
    if args.dry_run:
        return
    summary = summarize_final(rows)
    write_rows(args.output_dir / "final_test_three_seed_summary.csv", summary)
    write_json(args.output_dir / "final_selection.json", {
        "selected_projector": str(selected_projector),
        "selected_token_k": selected_k,
        "summary": summary,
    })
    print(f"Final summary: {args.output_dir / 'final_test_three_seed_summary.csv'}")


if __name__ == "__main__":
    main()
