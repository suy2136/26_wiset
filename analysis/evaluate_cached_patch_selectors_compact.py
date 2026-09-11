"""Sequential VP evaluation for cached K1/adaptive/K2 patch selectors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    ("cached_k1", "k1"),
    ("cached_adaptive", "adaptive"),
    ("cached_k2", "k2"),
)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--nbs-checkpoint", type=Path, required=True)
    result.add_argument("--projector-checkpoint", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--rank-budget", type=int, default=512)
    result.add_argument("--physical-rank", type=int, default=32)
    result.add_argument("--motion-threshold-deg", type=float, default=12.0)
    result.add_argument("--latency-warmup-steps", type=int, default=5)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def absolute(path):
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def common_command(args, model_path, result_dir):
    return [
        sys.executable, "run_plm.py", "--test",
        "--train-dataset", "Jin2022", "--test-dataset", "Jin2022",
        "--evaluation-split", "test",
        "--plm-type", "llama", "--plm-size", "base",
        "--device", args.device, "--device-out", args.device,
        "--fp16", "--rank", str(args.physical_rank), "--use-adalora",
        "--adalora-allocator", "nbs",
        "--adalora-rank-config",
        "configs/adalora_rank_config_llama7b_min2_max32.json",
        "--adalora-rank-budget", str(args.rank_budget),
        "--adalora-ema-beta", "0.9",
        "--adalora-shadow-update-policy", "legacy",
        "--adalora-allocation-interval", "10",
        "--experiment-tag", "nbs_v19",
        "--model-path", str(model_path), "--evaluation-tag", "best_ar",
        "--nbs-inference-mode", "compact",
        "--epochs", "4", "--bs", "1", "--grad-accum-steps", "32",
        "--lr", "0.0002", "--seed", "1", "--lora-seed", "1",
        "--data-seed", "1", "--measure-inference-latency",
        "--latency-warmup-steps", str(args.latency_warmup_steps),
        "--save-test-progress-per-steps", "500",
        "--latency-output-path", str(result_dir / "latency.json"),
        "--results-output-dir", str(result_dir),
    ]


def build_command(args, model_path, result_dir, policy=None,
                  compact_output_dir=None):
    command = common_command(args, model_path, result_dir)
    if compact_output_dir is not None:
        command.extend(["--nbs-compact-output-dir", str(compact_output_dir)])
    if policy is not None:
        command.extend([
            "--multimodal-mode", "cached-patch-selection",
            "--cached-patch-policy", policy,
            "--cached-patch-motion-threshold-deg",
            str(args.motion_threshold_deg),
            "--cached-patch-features-dir", str(args.cache_dir),
            "--cached-patch-cache-device", "model",
            "--cached-patch-preload",
            "--cached-patch-stats-output-path",
            str(result_dir / "selector_stats.json"),
            "--multimodal-projector-checkpoint",
            str(args.projector_checkpoint),
            "--inference-tag", f"cached_{policy}",
        ])
    return command


def aggregate_result(directory):
    result_files = sorted(
        path for path in directory.glob("*_results.csv")
        if "_partial_" not in path.name and "_per_sample_" not in path.name
    )
    if not result_files:
        raise FileNotFoundError(f"aggregate results CSV absent in {directory}")
    with result_files[-1].open(newline="", encoding="utf-8-sig") as stream:
        aggregate = next(
            row for row in csv.DictReader(stream)
            if int(float(row["video"])) == -1
            and int(float(row["user"])) == -1
        )
    latency = json.loads((directory / "latency.json").read_text())
    return {
        "mae": float(aggregate["mae"]),
        "rmse": float(aggregate["rmse"]),
        "latency_mean_ms": float(latency["mean_s"]) * 1000.0,
        "latency_p50_ms": float(latency["median_s"]) * 1000.0,
        "latency_p95_ms": float(latency["p95_s"]) * 1000.0,
    }


def run_case(command, directory, resume):
    metrics_path = directory / "metrics.json"
    if resume and metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=REPO_ROOT, stdout=log,
            stderr=subprocess.STDOUT, check=True,
        )
    metrics = aggregate_result(directory)
    stats_path = directory / "selector_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        metrics["selected_patches_mean"] = stats["selected_patches_mean"]
        metrics["visual_tokens_per_call"] = stats["visual_tokens_per_call"]
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped PNG plot", flush=True)
        return
    labels = [row["case"] for row in rows]
    colors = ["#3675B5", "#4E9F6D", "#E3A52B", "#B45A68"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(labels, [row["mae"] for row in rows], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_title("VP MAE (lower is better)")
    axes[0].set_xlabel("MAE (degrees)")
    axes[1].barh(
        labels, [row["latency_mean_ms"] for row in rows], color=colors
    )
    axes[1].invert_yaxis()
    axes[1].set_title("Mean inference latency (lower is better)")
    axes[1].set_xlabel("Milliseconds")
    fig.suptitle("NBS-v19 compact: cached patch-selector comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_inputs(args):
    required = (
        args.nbs_checkpoint / "adapter_model.bin",
        args.nbs_checkpoint / "modules_except_plm.bin",
        args.nbs_checkpoint / "nash_rank_allocator.pt",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required NBS input absent: {path}")
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


def main():
    args = parser().parse_args()
    for name in ("nbs_checkpoint", "projector_checkpoint", "cache_dir",
                 "output_dir"):
        setattr(args, name, absolute(getattr(args, name)))
    compact_dir = args.output_dir / "compact_checkpoint"
    compact_exists = (compact_dir / "compact_adapter.pt").is_file()
    baseline_model = compact_dir if compact_exists else args.nbs_checkpoint
    commands = [("pure_nbs_compact", None, build_command(
        args, baseline_model, args.output_dir / "pure_nbs_compact",
        compact_output_dir=None if compact_exists else compact_dir,
    ))]
    commands.extend(
        (name, policy, build_command(
            args, compact_dir, args.output_dir / name, policy=policy
        ))
        for name, policy in POLICIES
    )
    if args.dry_run:
        print(json.dumps([
            {"case": name, "policy": policy, "command": command}
            for name, policy, command in commands
        ], indent=2))
        return
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, policy, command in commands:
        directory = args.output_dir / name
        row = {"case": name, "policy": policy or "none", "status": "failed"}
        try:
            row.update(run_case(command, directory, args.resume))
            row["status"] = "complete"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        write_rows(args.output_dir / "comparison_results.csv", rows)
        print(
            f"[{name}] {row['status']} MAE={row.get('mae')} "
            f"latency={row.get('latency_mean_ms')}", flush=True,
        )
        if (name == "pure_nbs_compact"
                and not (compact_dir / "compact_adapter.pt").is_file()):
            raise RuntimeError("baseline did not produce compact checkpoint")
    write_rows(args.output_dir / "comparison_results.csv", rows)
    completed = [row for row in rows if row["status"] == "complete"]
    if completed:
        plot_rows(completed, args.output_dir / "cached_patch_overview.png")
    print(f"Saved: {args.output_dir / 'comparison_results.csv'}")


if __name__ == "__main__":
    main()
