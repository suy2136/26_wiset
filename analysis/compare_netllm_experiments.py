"""Compare the latest completed NBS-NetLLM and plain NetLLM runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("viewport_prediction/data/experiment_runs/netllm_vs_nbs"),
    )
    parser.add_argument("--nbs-run", type=Path)
    parser.add_argument("--plain-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def is_complete(run_dir: Path) -> bool:
    status_path = run_dir / "status.json"
    summary_path = run_dir / "figures" / "summary.json"
    if not status_path.exists() or not summary_path.exists():
        return False
    try:
        return json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def latest_complete(root: Path, variant: str) -> Path:
    variant_root = root / variant
    candidates = (
        sorted((path for path in variant_root.iterdir() if path.is_dir()), reverse=True)
        if variant_root.exists()
        else []
    )
    for candidate in candidates:
        if is_complete(candidate):
            return candidate
    raise FileNotFoundError(f"No completed {variant} experiment found under {variant_root}")


def load_summary(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "figures" / "summary.json").read_text(encoding="utf-8"))
    summary["run_dir"] = str(run_dir)
    return summary


def relative_change(nbs: float, plain: float) -> float | None:
    if plain == 0 or not math.isfinite(nbs) or not math.isfinite(plain):
        return None
    return (nbs - plain) / plain * 100.0


def latency_ms(summary: dict[str, Any], key: str) -> float | None:
    value = (summary.get("inference_latency") or {}).get(key)
    if value is None:
        return None
    value = float(value) * 1000.0
    return value if math.isfinite(value) else None


def plot_curve(axis: Any, summary: dict[str, Any], key: str, x_key: str, color: str) -> None:
    points = [point for point in summary.get(key, []) if math.isfinite(float(point["loss"]))]
    if points:
        axis.plot(
            [point[x_key] for point in points],
            [point["loss"] for point in points],
            label=summary["display_name"],
            linewidth=1.8,
            color=color,
        )


def main() -> None:
    args = parse_args()
    nbs_run = args.nbs_run or latest_complete(args.artifact_root, "nbs")
    plain_run = args.plain_run or latest_complete(args.artifact_root, "plain")
    if not is_complete(nbs_run) or not is_complete(plain_run):
        raise ValueError("Both experiment directories must be complete and contain figures/summary.json")

    nbs = load_summary(nbs_run)
    plain = load_summary(plain_run)
    summaries = [nbs, plain]
    output_dir = args.output_dir or args.artifact_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "nbs": nbs,
        "plain": plain,
        "relative_change_percent_nbs_vs_plain": {
            "mae": relative_change(float(nbs["aggregate_mae"]), float(plain["aggregate_mae"])),
            "rmse": relative_change(float(nbs["aggregate_rmse"]), float(plain["aggregate_rmse"])),
            "latency_mean": relative_change(
                latency_ms(nbs, "mean_s") or math.nan,
                latency_ms(plain, "mean_s") or math.nan,
            ),
            "latency_p95": relative_change(
                latency_ms(nbs, "p95_s") or math.nan,
                latency_ms(plain, "p95_s") or math.nan,
            ),
        },
    }
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "variant", "run_dir", "mae", "rmse", "final_train_loss",
                "best_valid_loss", "latency_mean_ms", "latency_median_ms",
                "latency_p95_ms",
            ),
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "variant": summary["display_name"],
                    "run_dir": summary["run_dir"],
                    "mae": summary["aggregate_mae"],
                    "rmse": summary["aggregate_rmse"],
                    "final_train_loss": summary.get("final_reported_train_loss"),
                    "best_valid_loss": summary.get("best_reported_valid_loss"),
                    "latency_mean_ms": latency_ms(summary, "mean_s"),
                    "latency_median_ms": latency_ms(summary, "median_s"),
                    "latency_p95_ms": latency_ms(summary, "p95_s"),
                }
            )

    labels = [summary["display_name"] for summary in summaries]
    colors = ["#d95f02", "#1b9e77"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    fig.suptitle("NBS-NetLLM vs NetLLM")
    axes[0, 0].bar(labels, [summary["aggregate_mae"] for summary in summaries], color=colors)
    axes[0, 0].set_title("Aggregate MAE (lower is better)")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 1].bar(labels, [summary["aggregate_rmse"] for summary in summaries], color=colors)
    axes[0, 1].set_title("Aggregate RMSE (lower is better)")
    axes[0, 1].grid(axis="y", alpha=0.25)

    mean_latencies = [latency_ms(summary, "mean_s") for summary in summaries]
    if all(value is not None for value in mean_latencies):
        bars = axes[0, 2].bar(labels, mean_latencies, color=colors)
        axes[0, 2].bar_label(bars, fmt="%.1f ms", padding=3)
        axes[0, 2].set_ylabel("Milliseconds per sample")
    else:
        axes[0, 2].text(0.5, 0.5, "Latency unavailable", ha="center", va="center")
    axes[0, 2].set_title("Mean inference latency (lower is better)")
    axes[0, 2].grid(axis="y", alpha=0.25)

    for summary, color in zip(summaries, colors):
        plot_curve(axes[1, 0], summary, "train_curve", "step", color)
        plot_curve(axes[1, 1], summary, "valid_curve", "index", color)
    axes[1, 0].set(title="Reported training loss", xlabel="Global step", ylabel="Loss")
    axes[1, 1].set(title="Validation loss", xlabel="Validation event", ylabel="Loss")
    for axis in axes[1]:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()

    latency_keys = ("mean_s", "median_s", "p95_s")
    latency_names = ("Mean", "Median", "P95")
    x_positions = list(range(len(latency_keys)))
    width = 0.36
    for index, (summary, color) in enumerate(zip(summaries, colors)):
        values = [latency_ms(summary, key) for key in latency_keys]
        if all(value is not None for value in values):
            offset = (index - 0.5) * width
            axes[1, 2].bar(
                [position + offset for position in x_positions],
                values,
                width=width,
                label=summary["display_name"],
                color=color,
            )
    axes[1, 2].set_xticks(x_positions, latency_names)
    axes[1, 2].set_title("Inference latency distribution")
    axes[1, 2].set_ylabel("Milliseconds per sample")
    axes[1, 2].grid(axis="y", alpha=0.25)
    if axes[1, 2].patches:
        axes[1, 2].legend()

    for axis in axes[0]:
        axis.tick_params(axis="x", labelrotation=10)

    figure_path = output_dir / "netllm_vs_nbs_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Compared NBS run: {nbs_run}")
    print(f"Compared plain run: {plain_run}")
    print(f"Comparison data saved at {json_path} and {csv_path}")
    print(f"Comparison figure saved at {figure_path}")


if __name__ == "__main__":
    main()
