"""Evaluate the tuned-projector VP full stack for seeds 1, 2, and 3 only."""

from __future__ import annotations

import argparse
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
from analysis.evaluate_vp_fixed_spec_patch_token_sweep import (
    SPEC_GAMMA,
    SPEC_THRESHOLD,
    full_stack_command,
)
from analysis.evaluate_vp_inference_module_pipeline import (
    set_option,
    trace_metrics,
    write_json,
)


SEEDS = (1, 2, 3)
CASE = "tuned_patch_token_spec_full_stack"
PATCH_CONFIG = {
    "policy": "gated-k1",
    "threshold": 6.0,
    "max_skip": 0,
    "projector_cache": True,
}
TOKEN_K = 8


def parser() -> argparse.ArgumentParser:
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


def projector_file(path: Path) -> Path:
    return path / "modules_except_plm.bin" if path.is_dir() else path


def validate_inputs(args) -> None:
    for name in (
        "compact_adapter.pt", "modules_except_plm.bin",
        "equivalence_report.json",
    ):
        path = args.compact_checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"compact checkpoint input absent: {path}")
    projector = projector_file(args.tuned_projector)
    if not projector.is_file():
        raise FileNotFoundError(f"tuned projector absent: {projector}")
    for video in (4, 8, 14, 18, 24, 25):
        path = args.cache_dir / f"video{video}_patch_features.pt"
        if not path.is_file():
            raise FileNotFoundError(f"test-video patch cache absent: {path}")


def apply_seed(command: list[str], seed: int) -> list[str]:
    set_option(command, "--seed", seed)
    set_option(command, "--lora-seed", 1)
    set_option(command, "--data-seed", seed)
    return command


def build_command(args, result_dir: Path, seed: int) -> list[str]:
    command = full_stack_command(
        args,
        args.compact_checkpoint,
        result_dir,
        PATCH_CONFIG,
        TOKEN_K,
    )
    return apply_seed(command, seed)


def summarize(rows: list[dict]) -> dict:
    complete = [row for row in rows if row.get("status") == "complete"]
    if {int(row["evaluation_seed"]) for row in complete} != set(SEEDS):
        raise RuntimeError("full stack does not have complete seeds 1, 2, 3")
    summary = {
        "case": CASE,
        "label": "NBS + tuned Patch + Token K=8 + Spec G=6/T=0.4",
        "seed_count": 3,
        "evaluation_seeds": "1,2,3",
    }
    for key in (
        "mae", "rmse", "latency_mean_ms", "latency_p50_ms",
        "latency_p95_ms", "mean_initial_token_count",
        "mean_selected_token_count", "mean_target_forward_count",
        "mean_token_reduction_percent", "draft_acceptance_rate",
        "selected_patches_mean", "visual_tokens_per_call",
    ):
        values = [float(row[key]) for row in complete if row.get(key) != ""]
        if len(values) == 3:
            summary[f"{key}_mean"] = statistics.mean(values)
            summary[f"{key}_std"] = statistics.stdev(values)
    if "mean_target_forward_count_mean" in summary:
        summary["target_forward_reduction_percent_vs_ar20"] = (
            1.0 - summary["mean_target_forward_count_mean"] / 20.0
        ) * 100.0
        summary["effective_ms_per_target_forward"] = (
            summary["latency_mean_ms_mean"]
            / summary["mean_target_forward_count_mean"]
        )
    return summary


def write_manifest(args) -> None:
    signature = {
        "compact_checkpoint": str(args.compact_checkpoint),
        "tuned_projector": str(args.tuned_projector),
        "cache_dir": str(args.cache_dir),
        "evaluation_split": "test",
        "evaluation_seeds": list(SEEDS),
        "lora_seed": 1,
        "patch": PATCH_CONFIG,
        "token_k": TOKEN_K,
        "spec_gamma": SPEC_GAMMA,
        "spec_threshold": SPEC_THRESHOLD,
    }
    path = args.output_dir / "manifest.json"
    if path.is_file() and args.resume:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError("resume manifest differs from current configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    write_json(path, {"signature": signature})


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    for name in (
        "compact_checkpoint", "tuned_projector", "cache_dir", "output_dir",
    ):
        setattr(args, name, absolute(getattr(args, name)))
    # full_stack_command expects this shared runner attribute.
    args.projector_checkpoint = args.tuned_projector
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        validate_inputs(args)
        write_manifest(args)

    rows = []
    for seed in SEEDS:
        result_dir = args.output_dir / f"seed_{seed}" / CASE
        command = build_command(args, result_dir, seed)
        if args.dry_run:
            print(json.dumps({
                "case": CASE,
                "seed": seed,
                "command": command,
            }, indent=2), flush=True)
            continue
        row = {
            "case": CASE,
            "evaluation_seed": seed,
            "lora_seed": 1,
            "data_seed": seed,
            "projector_checkpoint": str(args.tuned_projector),
            "status": "failed",
        }
        try:
            row.update(run_case(command, result_dir, args.resume))
            row.update(trace_metrics(result_dir))
            row["status"] = "complete"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        write_rows(args.output_dir / "per_seed_results.csv", rows)
        print(
            f"[{CASE} seed={seed}] {row['status']} "
            f"MAE={row.get('mae')} latency={row.get('latency_mean_ms')}",
            flush=True,
        )

    if args.dry_run:
        print("Dry run: 3 tuned-projector full-stack evaluations", flush=True)
        return
    failed = [
        f"seed{row['evaluation_seed']}" for row in rows
        if row["status"] != "complete"
    ]
    if failed:
        raise RuntimeError(f"failed full-stack evaluations: {', '.join(failed)}")
    summary = summarize(rows)
    write_rows(args.output_dir / "three_seed_summary.csv", [summary])
    write_json(args.output_dir / "three_seed_summary.json", summary)
    print(f"Per-seed results: {args.output_dir / 'per_seed_results.csv'}")
    print(f"Three-seed summary: {args.output_dir / 'three_seed_summary.csv'}")


if __name__ == "__main__":
    main()
