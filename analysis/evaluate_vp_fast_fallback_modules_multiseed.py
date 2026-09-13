"""Evaluate four VP module ablations under the compact fast/fallback path."""

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
    absolute,
    common_command,
    run_case,
    write_rows,
)
from analysis.evaluate_vp_fast_fallback_fullstack_multiseed import (
    PATCH,
    SPEC_GAMMA,
    SPEC_THRESHOLD,
    TOKEN_K,
    read_safety,
)
from analysis.evaluate_vp_inference_module_pipeline import (
    set_option,
    speculative_command,
    token_command,
    trace_metrics,
)


SEEDS = (1, 2, 3)
CASES = (
    ("pure_nbs_compact", "Pure NBS compact", "baseline"),
    (
        "patch_gated_k1_t6_skip0_cache",
        "NBS + Patch gated-k1/T6/skip0/cache",
        "patch",
    ),
    ("token_k8", "NBS + Token K=8", "token"),
    ("spec_g6_t0p4", "NBS + Spec G=6/T=0.4", "speculative"),
)


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
    if args.rank_budget != 512 or args.physical_rank != 32:
        raise ValueError("this comparison is fixed to budget=512, physical rank=32")
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
    missing = [
        video for video in (4, 8, 14, 18, 24, 25)
        if not (args.cache_dir / f"video{video}_patch_features.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"test patch caches missing: {missing}")


def command_for(args, kind, seed, result_dir):
    if kind == "baseline":
        command = common_command(args, args.compact_checkpoint, result_dir)
    elif kind == "patch":
        command = selector_command(
            args, args.compact_checkpoint, result_dir, PATCH
        )
    elif kind == "token":
        command = token_command(
            args, args.compact_checkpoint, result_dir, TOKEN_K
        )
    elif kind == "speculative":
        command = speculative_command(
            args,
            args.compact_checkpoint,
            result_dir,
            SPEC_GAMMA,
            SPEC_THRESHOLD,
        )
    else:
        raise ValueError(kind)
    set_option(command, "--seed", seed)
    set_option(command, "--lora-seed", 1)
    set_option(command, "--data-seed", seed)
    set_option(command, "--evaluation-rng-mode", "continuous")
    return command


def summarize(rows):
    summaries = []
    for case, label, _ in CASES:
        group = [
            row for row in rows
            if row.get("case") == case and row.get("status") == "complete"
        ]
        if {int(row["evaluation_seed"]) for row in group} != set(SEEDS):
            raise RuntimeError(f"incomplete three-seed result: {case}")
        item = {
            "case": case,
            "label": label,
            "seed_count": 3,
            "evaluation_seeds": "1,2,3",
            "evaluation_rng_mode": "continuous",
        }
        for metric in ("mae", "rmse", "latency_mean_ms"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std"] = statistics.stdev(values)
        for metric in (
            "fp16_prescaled_qk_fallback_calls",
            "fp32_attention_fallback_calls",
        ):
            item[f"{metric}_total"] = sum(
                int(row.get(metric, 0)) for row in group
            )
        summaries.append(item)
    baseline = summaries[0]
    for item in summaries:
        item["mae_change_percent_vs_nbs"] = (
            item["mae_mean"] / baseline["mae_mean"] - 1.0
        ) * 100.0
        item["latency_reduction_percent_vs_nbs"] = (
            1.0 - item["latency_mean_ms_mean"]
            / baseline["latency_mean_ms_mean"]
        ) * 100.0
    return summaries


def plot_summary(rows, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped summary plot", flush=True)
        return
    labels = [row["label"] for row in rows]
    colors = ["#3675B5", "#4E9F6D", "#E3A52B", "#B45A68"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(
        labels,
        [row["mae_mean"] for row in rows],
        xerr=[row["mae_std"] for row in rows],
        color=colors,
        capsize=4,
    )
    axes[0].invert_yaxis()
    axes[0].set_title("VP MAE, 3-seed mean")
    axes[0].set_xlabel("Degrees; lower is better")
    axes[1].barh(
        labels,
        [row["latency_mean_ms_mean"] for row in rows],
        xerr=[row["latency_mean_ms_std"] for row in rows],
        color=colors,
        capsize=4,
    )
    axes[1].invert_yaxis()
    axes[1].set_title("VP latency, 3-seed mean")
    axes[1].set_xlabel("Milliseconds; lower is better")
    figure.suptitle("VP compact fast-path module ablation")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_manifest(args):
    signature = {
        "compact_checkpoint": str(args.compact_checkpoint),
        "projector_checkpoint": str(args.projector_checkpoint),
        "cache_dir": str(args.cache_dir),
        "rank_budget": 512,
        "physical_rank": 32,
        "projector_epochs": 2,
        "patch": PATCH,
        "token_k": TOKEN_K,
        "spec_gamma": SPEC_GAMMA,
        "spec_threshold": SPEC_THRESHOLD,
        "cases": CASES,
        "evaluation_seeds": list(SEEDS),
        "evaluation_rng_mode": "continuous",
        "numeric_safety": "normal-fast-path; prescaled-QK retry on nonfinite",
    }
    path = args.output_dir / "manifest.json"
    if path.is_file() and args.resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("signature") != json.loads(json.dumps(signature)):
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
        (row["case"], int(row["evaluation_seed"]))
        for row in rows if row.get("status") == "complete"
    }
    for seed in SEEDS:
        for case, label, kind in CASES:
            if (case, seed) in completed:
                continue
            result_dir = args.output_dir / f"seed_{seed}" / case
            command = command_for(args, kind, seed, result_dir)
            if args.dry_run:
                print(json.dumps({"case": case, "seed": seed, "command": command}))
                continue
            row = {
                "case": case,
                "label": label,
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
                f"[{case} seed={seed}] {row['status']} "
                f"MAE={row.get('mae')} latency={row.get('latency_mean_ms')} "
                f"fallbacks={row.get('fp16_prescaled_qk_fallback_calls')}",
                flush=True,
            )
            if row["status"] != "complete":
                raise RuntimeError(row["error"])
    if args.dry_run:
        print("Dry run: 4 configurations x 3 seeds = 12 evaluations", flush=True)
        return
    summaries = summarize(rows)
    write_rows(args.output_dir / "three_seed_summary.csv", summaries)
    (args.output_dir / "three_seed_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    plot_summary(summaries, args.output_dir / "three_seed_summary.png")
    print(f"Per-seed results: {rows_path}")
    print(f"Three-seed summary: {args.output_dir / 'three_seed_summary.csv'}")


if __name__ == "__main__":
    main()
