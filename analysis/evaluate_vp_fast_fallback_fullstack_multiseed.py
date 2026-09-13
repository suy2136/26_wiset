"""Evaluate one fixed VP full stack with compact fast-path safety fallback."""

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
    absolute,
    run_case,
    write_rows,
)
from analysis.evaluate_vp_fixed_spec_patch_token_sweep import full_stack_command
from analysis.evaluate_vp_inference_module_pipeline import (
    set_option,
    trace_metrics,
)


SEEDS = (1, 2, 3)
PATCH = {
    "policy": "gated-k1",
    "threshold": 6.0,
    "max_skip": 0,
    "projector_cache": True,
}
TOKEN_K = 8
SPEC_GAMMA = 6
SPEC_THRESHOLD = 0.4
CASE = "fast_fallback_patch_gated_k1_t6_token_k8_spec_g6_t0p4"


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--compact-checkpoint", type=Path, required=True)
    result.add_argument("--projector-checkpoint", type=Path, required=True)
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


def validate_inputs(args):
    for name in ("compact_adapter.pt", "modules_except_plm.bin"):
        if not (args.compact_checkpoint / name).is_file():
            raise FileNotFoundError(args.compact_checkpoint / name)
    projector = args.projector_checkpoint
    if projector.is_dir():
        candidates = (
            projector / "best_multimodal_projector.pth",
            projector / "modules_except_plm.bin",
        )
        if not any(path.is_file() for path in candidates):
            raise FileNotFoundError(f"projector checkpoint absent: {projector}")
    elif not projector.is_file():
        raise FileNotFoundError(projector)
    required_videos = (4, 8, 14, 18, 24, 25)
    missing = [
        video for video in required_videos
        if not (args.cache_dir / f"video{video}_patch_features.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"test patch caches missing: {missing}")


def command_for(args, seed, result_dir):
    # full_stack_command reads these common runner attributes.
    args.projector_checkpoint = args.projector_checkpoint
    command = full_stack_command(
        args,
        args.compact_checkpoint,
        result_dir,
        PATCH,
        TOKEN_K,
    )
    set_option(command, "--seed", seed)
    set_option(command, "--lora-seed", 1)
    set_option(command, "--data-seed", seed)
    set_option(command, "--evaluation-rng-mode", "continuous")
    return command


def read_safety(result_dir):
    path = result_dir / "vp_numeric_safety.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows):
    complete = [row for row in rows if row.get("status") == "complete"]
    if {int(row["evaluation_seed"]) for row in complete} != set(SEEDS):
        raise RuntimeError("full-stack evaluation does not contain seeds 1, 2, 3")
    summary = {
        "case": CASE,
        "seed_count": 3,
        "evaluation_seeds": "1,2,3",
        "evaluation_rng_mode": "continuous",
        "rank_budget": 512,
        "physical_rank": 32,
        "projector_epochs": 2,
        "patch": "gated-k1/T6/skip0/cache",
        "token": "K=8",
        "speculative": "G=6/T=0.4",
    }
    for metric in ("mae", "rmse", "latency_mean_ms"):
        values = [float(row[metric]) for row in complete]
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_std"] = statistics.stdev(values)
    for metric in (
        "fp16_prescaled_qk_fallback_calls",
        "fp32_attention_fallback_calls",
    ):
        summary[f"{metric}_total"] = sum(
            int(row.get(metric, 0)) for row in complete
        )
    return summary


def write_manifest(args):
    signature = {
        "compact_checkpoint": str(args.compact_checkpoint),
        "projector_checkpoint": str(args.projector_checkpoint),
        "cache_dir": str(args.cache_dir),
        "rank_budget": args.rank_budget,
        "physical_rank": args.physical_rank,
        "projector_epochs": 2,
        "patch": PATCH,
        "token_k": TOKEN_K,
        "spec_gamma": SPEC_GAMMA,
        "spec_threshold": SPEC_THRESHOLD,
        "evaluation_seeds": list(SEEDS),
        "evaluation_rng_mode": "continuous",
        "numeric_safety": "normal-fast-path; prescaled-QK retry on nonfinite",
    }
    path = args.output_dir / "manifest.json"
    if path.is_file() and args.resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError("resume manifest differs from current configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    path.write_text(
        json.dumps({"signature": signature}, indent=2), encoding="utf-8"
    )


def main(argv=None):
    args = parser().parse_args(argv)
    for name in (
        "compact_checkpoint", "projector_checkpoint", "cache_dir", "output_dir",
    ):
        setattr(args, name, absolute(getattr(args, name)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        validate_inputs(args)
        write_manifest(args)

    rows_path = args.output_dir / "per_seed_results.csv"
    rows = []
    if args.resume and rows_path.is_file():
        with rows_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    completed = {
        int(row["evaluation_seed"])
        for row in rows if row.get("status") == "complete"
    }
    for seed in SEEDS:
        if seed in completed:
            continue
        result_dir = args.output_dir / f"seed_{seed}" / CASE
        command = command_for(args, seed, result_dir)
        if args.dry_run:
            print(json.dumps({"seed": seed, "command": command}), flush=True)
            continue
        row = {
            "case": CASE,
            "evaluation_seed": seed,
            "lora_seed": 1,
            "data_seed": seed,
            "status": "failed",
        }
        try:
            row.update(run_case(command, result_dir, args.resume))
            row.update(trace_metrics(result_dir))
            row.update(read_safety(result_dir))
            row["status"] = "complete"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        write_rows(rows_path, rows)
        print(
            f"[seed {seed}] {row['status']} MAE={row.get('mae')} "
            f"latency={row.get('latency_mean_ms')} "
            f"fallbacks={row.get('fp16_prescaled_qk_fallback_calls')}",
            flush=True,
        )
        if row["status"] != "complete":
            raise RuntimeError(row["error"])
    if args.dry_run:
        return
    result = summarize(rows)
    write_rows(args.output_dir / "three_seed_summary.csv", [result])
    (args.output_dir / "three_seed_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Per-seed results: {rows_path}")
    print(f"Three-seed summary: {args.output_dir / 'three_seed_summary.csv'}")


if __name__ == "__main__":
    main()
