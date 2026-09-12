"""Three-seed VP comparison with per-video-user RNG isolation."""

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

from analysis.evaluate_cached_patch_selectors_compact import (
    absolute, common_command, run_case, write_rows,
)
from analysis.evaluate_cached_patch_followups_compact import selector_command
from analysis.evaluate_vp_fixed_spec_patch_token_sweep import full_stack_command
from analysis.evaluate_vp_inference_module_pipeline import (
    set_option, token_command, trace_metrics, write_json,
)


SEEDS = (1, 2, 3)
PATCH_CONFIG = {
    "policy": "gated-k1", "threshold": 6.0,
    "max_skip": 0, "projector_cache": True,
}
TOKEN_K = 8
CONFIGURATIONS = (
    ("pure_nbs_compact", "NBS compact only", "baseline"),
    ("tuned_patch_gated_k1_t6_skip0_cache", "+ tuned Patch", "patch"),
    ("token_recent_k8", "+ Token K=8", "token"),
    (
        "tuned_patch_token_spec_full_stack",
        "+ tuned Patch + Token K=8 + Spec G=6/T=0.4",
        "full_stack",
    ),
)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--compact-checkpoint", type=Path, required=True)
    result.add_argument("--tuned-projector", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--rank-budget", type=int, default=512)
    result.add_argument("--physical-rank", type=int, default=32)
    result.add_argument("--latency-warmup-steps", type=int, default=5)
    result.add_argument("--projector-cache-max-entries", type=int, default=512)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def projector_file(path):
    return path / "modules_except_plm.bin" if path.is_dir() else path


def validate_inputs(args):
    for name in ("compact_adapter.pt", "modules_except_plm.bin"):
        path = args.compact_checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"compact checkpoint absent: {path}")
    if not projector_file(args.tuned_projector).is_file():
        raise FileNotFoundError(f"tuned projector absent: {args.tuned_projector}")
    for video in (4, 8, 14, 18, 24, 25):
        path = args.cache_dir / f"video{video}_patch_features.pt"
        if not path.is_file():
            raise FileNotFoundError(f"test-video patch cache absent: {path}")


def build_command(args, kind, result_dir, seed):
    # Shared builders retain all historical VP model/evaluation parameters.
    if kind == "baseline":
        command = common_command(args, args.compact_checkpoint, result_dir)
    elif kind == "patch":
        command = selector_command(
            args, args.compact_checkpoint, result_dir, PATCH_CONFIG
        )
    elif kind == "token":
        command = token_command(
            args, args.compact_checkpoint, result_dir, TOKEN_K
        )
    elif kind == "full_stack":
        command = full_stack_command(
            args, args.compact_checkpoint, result_dir, PATCH_CONFIG, TOKEN_K
        )
    else:
        raise ValueError(f"unknown configuration kind: {kind}")
    set_option(command, "--seed", seed)
    set_option(command, "--lora-seed", 1)
    set_option(command, "--data-seed", seed)
    set_option(command, "--evaluation-rng-mode", "per-episode")
    return command


def summarize(rows):
    summaries = []
    for case, label, _ in CONFIGURATIONS:
        group = sorted(
            (row for row in rows if row["case"] == case),
            key=lambda row: int(row["evaluation_seed"]),
        )
        if [int(row["evaluation_seed"]) for row in group] != list(SEEDS):
            raise RuntimeError(f"{case} does not have seeds 1, 2, 3")
        summary = {
            "case": case, "label": label, "seed_count": 3,
            "evaluation_seeds": "1,2,3",
            "evaluation_rng_mode": "per-episode",
        }
        for metric in (
            "mae", "rmse", "latency_mean_ms", "latency_p50_ms",
            "latency_p95_ms", "mean_initial_token_count",
            "mean_selected_token_count", "mean_target_forward_count",
            "mean_token_reduction_percent", "draft_acceptance_rate",
            "selected_patches_mean", "visual_tokens_per_call",
        ):
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
        "compact_checkpoint": str(args.compact_checkpoint),
        "tuned_projector": str(args.tuned_projector),
        "cache_dir": str(args.cache_dir),
        "evaluation_rng_mode": "per-episode",
        "evaluation_seeds": list(SEEDS),
        "patch": PATCH_CONFIG,
        "token_k": TOKEN_K,
        "spec_gamma": 6,
        "spec_threshold": 0.4,
        "configurations": [item[0] for item in CONFIGURATIONS],
    }
    path = args.output_dir / "manifest.json"
    if path.is_file() and args.resume:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError("resume manifest differs from this configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    write_json(path, {"signature": signature})


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("compact_checkpoint", "tuned_projector", "cache_dir", "output_dir"):
        setattr(args, name, absolute(getattr(args, name)))
    args.projector_checkpoint = args.tuned_projector
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        validate_inputs(args)
        write_manifest(args)

    rows = []
    runs_path = args.output_dir / "per_seed_results.csv"
    if args.resume and runs_path.is_file():
        with runs_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    completed = {
        (row["case"], int(row["evaluation_seed"]))
        for row in rows if row.get("status") == "complete"
    }
    for seed in SEEDS:
        for case, label, kind in CONFIGURATIONS:
            if (case, seed) in completed:
                print(f"[{case} seed={seed}] already complete; skipping", flush=True)
                continue
            result_dir = args.output_dir / f"seed_{seed}" / case
            command = build_command(args, kind, result_dir, seed)
            if args.dry_run:
                print(json.dumps({"case": case, "seed": seed, "command": command}))
                continue
            row = {
                "case": case, "label": label, "evaluation_seed": seed,
                "lora_seed": 1, "data_seed": seed,
                "evaluation_rng_mode": "per-episode", "status": "failed",
            }
            try:
                row.update(run_case(command, result_dir, args.resume))
                row.update(trace_metrics(result_dir))
                row["status"] = "complete"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            write_rows(runs_path, rows)
            print(
                f"[{case} seed={seed}] {row['status']} "
                f"MAE={row.get('mae')} latency={row.get('latency_mean_ms')}",
                flush=True,
            )
            if row["status"] != "complete":
                raise RuntimeError(row["error"])
    if args.dry_run:
        return
    summary = summarize(rows)
    write_rows(args.output_dir / "three_seed_summary.csv", summary)
    write_json(args.output_dir / "three_seed_summary.json", summary)
    print(f"Per-seed results: {runs_path}")
    print(f"Three-seed summary: {args.output_dir / 'three_seed_summary.csv'}")


if __name__ == "__main__":
    main()
