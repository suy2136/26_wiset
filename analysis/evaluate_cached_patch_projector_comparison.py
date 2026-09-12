"""Compare the original and a tuned VP cached-patch projector over 3 seeds."""

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
    run_case,
    write_rows,
)
from analysis.evaluate_vp_inference_module_pipeline import set_option


SEEDS = (1, 2, 3)
PATCH_CONFIG = {
    "policy": "gated-k1",
    "threshold": 6.0,
    "max_skip": 0,
    "projector_cache": True,
}
PROJECTORS = (
    ("original_projector", "Original projector"),
    ("tuned_projector", "K1 1-epoch tuned projector"),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--nbs-checkpoint", type=Path, required=True)
    result.add_argument("--original-projector", type=Path, required=True)
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
    required_nbs = (
        "adapter_model.bin", "modules_except_plm.bin", "nash_rank_allocator.pt",
    )
    for name in required_nbs:
        path = args.nbs_checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"required NBS input absent: {path}")
    for path in (args.original_projector, args.tuned_projector):
        resolved = projector_file(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"projector checkpoint absent: {resolved}")
    for video in (4, 8, 14, 18, 24, 25):
        path = args.cache_dir / f"video{video}_patch_features.pt"
        if not path.is_file():
            raise FileNotFoundError(f"test-video patch cache absent: {path}")


def apply_seed(command: list[str], seed: int) -> list[str]:
    set_option(command, "--seed", seed)
    set_option(command, "--lora-seed", 1)
    set_option(command, "--data-seed", seed)
    return command


def build_command(args, model_path: Path, projector: Path,
                  result_dir: Path, seed: int,
                  compact_output_dir: Path | None = None) -> list[str]:
    original = args.projector_checkpoint
    args.projector_checkpoint = projector
    try:
        command = selector_command(
            args, model_path, result_dir, PATCH_CONFIG
        )
    finally:
        args.projector_checkpoint = original
    if compact_output_dir is not None:
        command.extend(["--nbs-compact-output-dir", str(compact_output_dir)])
    return apply_seed(command, seed)


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for case, label in PROJECTORS:
        selected = [
            row for row in rows
            if row["case"] == case and row["status"] == "complete"
        ]
        if {int(row["evaluation_seed"]) for row in selected} != set(SEEDS):
            raise RuntimeError(f"{case} does not have complete seeds 1, 2, 3")
        item = {
            "case": case,
            "label": label,
            "seed_count": 3,
            "evaluation_seeds": "1,2,3",
        }
        for key in (
            "mae", "rmse", "latency_mean_ms", "latency_p50_ms",
            "latency_p95_ms", "selected_patches_mean",
            "visual_tokens_per_call",
        ):
            values = [float(row[key]) for row in selected if row.get(key) != ""]
            if len(values) == 3:
                item[f"{key}_mean"] = statistics.mean(values)
                item[f"{key}_std"] = statistics.stdev(values)
        summary.append(item)
    baseline = summary[0]
    for item in summary:
        item["mae_change_percent_vs_original"] = (
            item["mae_mean"] / baseline["mae_mean"] - 1.0
        ) * 100.0
        item["latency_reduction_percent_vs_original"] = (
            1.0 - item["latency_mean_ms_mean"]
            / baseline["latency_mean_ms_mean"]
        ) * 100.0
    return summary


def plot_summary(rows: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped PNG plot", flush=True)
        return
    labels = [row["label"] for row in rows]
    colors = ["#6F63B6", "#3675B5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(
        labels, [row["mae_mean"] for row in rows],
        xerr=[row["mae_std"] for row in rows], color=colors, capsize=4,
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Test MAE (lower is better)")
    axes[0].set_xlabel("Degrees, 3-seed mean +/- sample SD")
    axes[1].barh(
        labels, [row["latency_mean_ms_mean"] for row in rows],
        xerr=[row["latency_mean_ms_std"] for row in rows],
        color=colors, capsize=4,
    )
    axes[1].invert_yaxis()
    axes[1].set_title("Mean inference latency (lower is better)")
    axes[1].set_xlabel("Milliseconds, 3-seed mean +/- sample SD")
    fig.suptitle("VP cached-patch projector: original vs K1 tuned")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_manifest(args) -> None:
    signature = {
        "nbs_checkpoint": str(args.nbs_checkpoint),
        "original_projector": str(args.original_projector),
        "tuned_projector": str(args.tuned_projector),
        "cache_dir": str(args.cache_dir),
        "patch_config": PATCH_CONFIG,
        "evaluation_split": "test",
        "evaluation_seeds": list(SEEDS),
        "lora_seed": 1,
    }
    path = args.output_dir / "manifest.json"
    if path.is_file() and args.resume:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError("resume manifest differs from current configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    path.write_text(json.dumps({"signature": signature}, indent=2),
                    encoding="utf-8")


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    for name in (
        "nbs_checkpoint", "original_projector", "tuned_projector",
        "cache_dir", "output_dir",
    ):
        setattr(args, name, absolute(getattr(args, name)))
    args.projector_checkpoint = args.original_projector
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact_dir = args.output_dir / "compact_checkpoint"

    if not args.dry_run:
        validate_inputs(args)
        write_manifest(args)

    rows = []
    projectors = {
        "original_projector": args.original_projector,
        "tuned_projector": args.tuned_projector,
    }
    for seed in SEEDS:
        for case, label in PROJECTORS:
            result_dir = args.output_dir / f"seed_{seed}" / case
            compact_exists = (compact_dir / "compact_adapter.pt").is_file()
            model_path = compact_dir if compact_exists else args.nbs_checkpoint
            compact_output = None if compact_exists else compact_dir
            command = build_command(
                args, model_path, projectors[case], result_dir, seed,
                compact_output_dir=compact_output,
            )
            if args.dry_run:
                print(json.dumps({
                    "case": case, "seed": seed, "command": command,
                }, indent=2), flush=True)
                continue
            row = {
                "case": case,
                "label": label,
                "evaluation_seed": seed,
                "lora_seed": 1,
                "projector_checkpoint": str(projectors[case]),
                "status": "failed",
            }
            try:
                row.update(run_case(command, result_dir, args.resume))
                row["status"] = "complete"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            write_rows(args.output_dir / "per_seed_results.csv", rows)
            print(
                f"[{case} seed={seed}] {row['status']} "
                f"MAE={row.get('mae')} latency={row.get('latency_mean_ms')}",
                flush=True,
            )
            if not (compact_dir / "compact_adapter.pt").is_file():
                raise RuntimeError("first evaluation did not create compact checkpoint")

    if args.dry_run:
        print("Dry run: 6 test evaluations", flush=True)
        return
    failed = [
        f"{row['case']}:seed{row['evaluation_seed']}"
        for row in rows if row["status"] != "complete"
    ]
    if failed:
        raise RuntimeError(f"failed evaluations: {', '.join(failed)}")
    summary = summarize(rows)
    write_rows(args.output_dir / "three_seed_summary.csv", summary)
    (args.output_dir / "three_seed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_summary(summary, args.output_dir / "projector_comparison.png")
    print(f"Per-seed results: {args.output_dir / 'per_seed_results.csv'}")
    print(f"Three-seed summary: {args.output_dir / 'three_seed_summary.csv'}")


if __name__ == "__main__":
    main()
