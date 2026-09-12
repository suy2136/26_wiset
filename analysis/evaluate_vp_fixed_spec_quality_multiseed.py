"""Confirm NBS compact and the quality-first VP full stack over three seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.evaluate_cached_patch_selectors_compact import (
    absolute,
    common_command,
    run_case,
    write_rows,
)
from analysis.evaluate_cached_patch_followups_compact import selector_command
from analysis.evaluate_vp_fixed_spec_patch_token_sweep import (
    SPEC_GAMMA,
    SPEC_THRESHOLD,
    full_stack_command,
)
from analysis.evaluate_vp_inference_module_pipeline import (
    finite,
    set_option,
    speculative_command,
    token_command,
    trace_metrics,
    write_json,
)


SEEDS = (1, 2, 3)
QUALITY_CASE = "full_gated_k1_t6_skip0_cache_token_k8_spec_g6_t0p4"
PATCH_CASE = "patch_gated_k1_t6_skip0_cache"
QUALITY_PATCH = {
    "policy": "gated-k1",
    "threshold": 6.0,
    "max_skip": 0,
    "projector_cache": True,
}
TOKEN_K = 8
CONFIGURATIONS = (
    {
        "case": "pure_nbs_compact", "family": "baseline",
        "label": "NBS compact only", "kind": "baseline",
        "seed1_source": "compact",
    },
    {
        "case": PATCH_CASE, "family": "patch",
        "label": "+ Patch gated-k1/T6/skip0/cache", "kind": "patch",
        "seed1_source": None,
    },
    {
        "case": "token_recent_k8", "family": "token",
        "label": "+ Token K=8", "kind": "token",
        "seed1_source": "compact",
    },
    {
        "case": "spec_g6_t0p4", "family": "speculative",
        "label": "+ Spec G=6/T=0.4", "kind": "speculative",
        "seed1_source": "compact",
    },
    {
        "case": QUALITY_CASE, "family": "full_stack",
        "label": "+ Patch + Token + Spec", "kind": "full_stack",
        "seed1_source": "fixed",
    },
)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--seed1-run-dir", type=Path, required=True)
    result.add_argument("--compact-run-dir", type=Path, required=True)
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


def load_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def seed1_row(args, config):
    source_kind = config["seed1_source"]
    if source_kind is None:
        return None
    source = (
        args.compact_run_dir / "vp_module_sweep_results.csv"
        if source_kind == "compact"
        else args.seed1_run_dir / "fixed_spec_patch_token_sweep.csv"
    )
    matches = [
        row for row in load_csv(source)
        if row.get("case") == config["case"]
        and row.get("status") == "complete"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"seed-1 source must contain one complete {config['case']}: {source}"
        )
    row = dict(matches[0])
    for key in (
        "mae", "rmse", "latency_mean_ms", "latency_p50_ms",
        "latency_p95_ms",
    ):
        value = finite(row.get(key))
        if value is not None:
            row[key] = value
    row.update({
        "case": config["case"],
        "family": config["family"],
        "label": config["label"],
        "evaluation_seed": 1,
        "source": "reused_seed1",
        "status": "complete",
    })
    return row


def validate_inputs(args):
    compact = args.compact_run_dir / "compact_checkpoint"
    for name in ("compact_adapter.pt", "modules_except_plm.bin"):
        if not (compact / name).is_file():
            raise FileNotFoundError(f"compact checkpoint input absent: {compact / name}")
    projector = (
        args.projector_checkpoint / "modules_except_plm.bin"
        if args.projector_checkpoint.is_dir() else args.projector_checkpoint
    )
    if not projector.is_file():
        raise FileNotFoundError(f"projector checkpoint absent: {projector}")
    for video in (4, 8, 14, 18, 24, 25):
        path = args.cache_dir / f"video{video}_patch_features.pt"
        if not path.is_file():
            raise FileNotFoundError(f"test-video patch cache absent: {path}")
    for source in (
        args.seed1_run_dir / "fixed_spec_patch_token_sweep.csv",
        args.compact_run_dir / "vp_module_sweep_results.csv",
    ):
        if not source.is_file():
            raise FileNotFoundError(f"seed-1 result absent: {source}")
    return compact


def apply_evaluation_seed(command, seed):
    set_option(command, "--seed", seed)
    set_option(command, "--lora-seed", 1)
    set_option(command, "--data-seed", seed)
    return command


def evaluation_command(args, compact, result_dir, seed, kind):
    if kind == "baseline":
        command = common_command(args, compact, result_dir)
    elif kind == "patch":
        command = selector_command(
            args, compact, result_dir, QUALITY_PATCH
        )
    elif kind == "token":
        command = token_command(args, compact, result_dir, TOKEN_K)
    elif kind == "speculative":
        command = speculative_command(
            args, compact, result_dir, SPEC_GAMMA, SPEC_THRESHOLD
        )
    elif kind == "full_stack":
        command = full_stack_command(
            args, compact, result_dir, QUALITY_PATCH, TOKEN_K
        )
    else:
        raise ValueError(f"unknown evaluation kind: {kind}")
    return apply_evaluation_seed(command, seed)


def run_one(args, case, family, label, seed, command):
    result_dir = args.output_dir / f"seed_{seed}" / case
    row = {
        "case": case,
        "family": family,
        "label": label,
        "evaluation_seed": seed,
        "lora_seed": 1,
        "data_seed": seed,
        "source": "evaluated",
        "status": "failed",
    }
    if args.dry_run:
        row.update({"status": "dry_run", "command": command})
        print(f"[{case} seed={seed}] dry-run", flush=True)
        return row
    try:
        row.update(run_case(command, result_dir, args.resume))
        row.update(trace_metrics(result_dir))
        row["status"] = "complete"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    print(
        f"[{case} seed={seed}] {row['status']} MAE={row.get('mae')} "
        f"latency={row.get('latency_mean_ms')}", flush=True,
    )
    return row


def summarize(rows):
    output = []
    for config in CONFIGURATIONS:
        case = config["case"]
        selected = [
            row for row in rows
            if row["case"] == case and row["status"] == "complete"
        ]
        if {int(row["evaluation_seed"]) for row in selected} != set(SEEDS):
            raise RuntimeError(f"{case} does not have complete seeds 1, 2, 3")
        item = {
            "case": case,
            "label": config["label"],
            "seed_count": len(selected),
            "evaluation_seeds": "1,2,3",
        }
        for source, target in (
            ("mae", "mae"),
            ("rmse", "rmse"),
            ("latency_mean_ms", "latency_mean_ms"),
        ):
            values = [float(row[source]) for row in selected]
            item[f"{target}_mean"] = statistics.mean(values)
            item[f"{target}_std"] = statistics.stdev(values)
        output.append(item)
    baseline = output[0]
    for item in output:
        item["mae_change_percent_vs_nbs"] = (
            item["mae_mean"] / baseline["mae_mean"] - 1.0
        ) * 100.0
        item["latency_reduction_percent_vs_nbs"] = (
            1.0 - item["latency_mean_ms_mean"]
            / baseline["latency_mean_ms_mean"]
        ) * 100.0
    return output


def plot_summary(rows, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped summary PNG", flush=True)
        return
    labels = [row["label"] for row in rows]
    colors = ["#3274B4", "#4B9B69", "#DF9F2D", "#B65A6A", "#7554B8"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].barh(
        labels, [row["mae_mean"] for row in rows],
        xerr=[row["mae_std"] for row in rows], color=colors, capsize=4,
    )
    axes[0].invert_yaxis()
    axes[0].set_title("3-seed VP MAE (lower is better)")
    axes[0].set_xlabel("MAE (degrees), mean +/- sample std")
    axes[1].barh(
        labels, [row["latency_mean_ms_mean"] for row in rows],
        xerr=[row["latency_mean_ms_std"] for row in rows],
        color=colors, capsize=4,
    )
    axes[1].invert_yaxis()
    axes[1].set_title("3-seed mean inference latency")
    axes[1].set_xlabel("Milliseconds, mean +/- sample std")
    fig.suptitle("VP inference-module ablation: 3-seed mean")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_manifest(args):
    path = args.output_dir / "manifest.json"
    signature = {
        "seed1_run_dir": str(args.seed1_run_dir),
        "compact_run_dir": str(args.compact_run_dir),
        "projector_checkpoint": str(args.projector_checkpoint),
        "cache_dir": str(args.cache_dir),
        "evaluation_seeds": list(SEEDS),
        "lora_seed": 1,
        "quality_case": QUALITY_CASE,
        "patch_case": PATCH_CASE,
        "patch": QUALITY_PATCH,
        "token_k": TOKEN_K,
        "spec_gamma": SPEC_GAMMA,
        "spec_threshold": SPEC_THRESHOLD,
        "configurations": CONFIGURATIONS,
    }
    if path.is_file() and args.resume:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError("resume manifest differs from current configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    write_json(path, {"signature": signature})


def main(argv=None):
    args = parser().parse_args(argv)
    for name in (
        "seed1_run_dir", "compact_run_dir", "projector_checkpoint",
        "cache_dir", "output_dir",
    ):
        setattr(args, name, absolute(getattr(args, name)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact = args.compact_run_dir / "compact_checkpoint"
    if not args.dry_run:
        compact = validate_inputs(args)
        write_manifest(args)
        rows = [
            row for row in (seed1_row(args, config) for config in CONFIGURATIONS)
            if row is not None
        ]
    else:
        rows = []

    for seed in SEEDS:
        for config in CONFIGURATIONS:
            if seed == 1 and config["seed1_source"] is not None:
                continue
            case, family, label = (
                config["case"], config["family"], config["label"]
            )
            result_dir = args.output_dir / f"seed_{seed}" / case
            command = evaluation_command(
                args, compact, result_dir, seed, config["kind"]
            )
            rows.append(run_one(args, case, family, label, seed, command))
            if not args.dry_run:
                write_rows(args.output_dir / "per_seed_results.csv", rows)
    if args.dry_run:
        print(
            "Dry run: 11 evaluations (Patch seeds 1,2,3; four other "
            "cases seeds 2,3)", flush=True,
        )
        return

    failed = [
        f"{row['case']}:seed{row['evaluation_seed']}"
        for row in rows if row["status"] != "complete"
    ]
    if failed:
        raise RuntimeError(f"failed evaluations: {', '.join(failed)}")
    summary = summarize(rows)
    write_rows(args.output_dir / "three_seed_summary.csv", summary)
    write_json(args.output_dir / "three_seed_summary.json", summary)
    plot_summary(summary, args.output_dir / "three_seed_summary.png")
    print(f"Per-seed results: {args.output_dir / 'per_seed_results.csv'}")
    print(f"Three-seed summary: {args.output_dir / 'three_seed_summary.csv'}")


if __name__ == "__main__":
    main()
