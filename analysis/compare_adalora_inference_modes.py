"""Compare direct AR and selector+speculative inference for one AdaLoRA checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_case(directory: Path, name: str, direct: bool) -> dict:
    summary = read_json(directory / "figures" / "summary.json")
    latency = read_json(directory / "latency.json")
    trace_path = directory / "inference_trace.json"
    trace = read_json(trace_path) if trace_path.is_file() else {}
    return {
        "mode": name,
        "mae": summary["aggregate_mae"],
        "rmse": summary["aggregate_rmse"],
        "latency_mean_ms": latency["mean_s"] * 1000.0,
        "latency_median_ms": latency["median_s"] * 1000.0,
        "latency_p95_ms": latency["p95_s"] * 1000.0,
        "peak_memory_mb": latency.get("peak_memory_mb"),
        "target_forward_count": 20.0 if direct else trace.get("mean_target_forward_count"),
        "token_reduction_percent": 0.0 if direct else trace.get("mean_token_reduction_percent"),
        "draft_acceptance_rate": None if direct else trace.get("draft_acceptance_rate"),
    }


def plot_bar(axis, labels, values, title, ylabel, color) -> None:
    bars = axis.bar(labels, values, color=color, edgecolor="white")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    upper = max(values) * 1.18 if max(values) > 0 else 1.0
    axis.set_ylim(0, upper)
    for patch, value in zip(bars, values):
        axis.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + upper * 0.015,
            f"{value:.2f}",
            ha="center",
            fontsize=9,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--full-stack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        load_case(args.direct_dir, "Direct AR", direct=True),
        load_case(args.full_stack_dir, "Selector + Speculative", direct=False),
    ]
    baseline = rows[0]
    for row in rows:
        for metric in ("mae", "rmse", "latency_mean_ms", "latency_p95_ms"):
            row[f"{metric}_change_percent_vs_direct"] = (
                100.0 * (row[metric] - baseline[metric]) / baseline[metric]
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "adalora_inference_modes.csv"
    json_path = args.output_dir / "adalora_inference_modes.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    labels = [row["mode"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plot_bar(axes[0, 0], labels, [r["mae"] for r in rows], "MAE", "Degrees", "#2a9d8f")
    plot_bar(axes[0, 1], labels, [r["rmse"] for r in rows], "RMSE", "Degrees", "#6c6fb3")
    plot_bar(
        axes[0, 2], labels, [r["latency_mean_ms"] for r in rows],
        "Mean inference latency", "ms / sample", "#e76f51",
    )
    plot_bar(
        axes[1, 0], labels, [r["latency_p95_ms"] for r in rows],
        "p95 inference latency", "ms / sample", "#f4a261",
    )
    plot_bar(
        axes[1, 1], labels, [r["target_forward_count"] for r in rows],
        "Target-model forward count", "Calls / sample", "#7b2cbf",
    )
    plot_bar(
        axes[1, 2], labels, [r["peak_memory_mb"] for r in rows],
        "Peak allocated GPU memory", "MiB", "#bc6c25",
    )
    fig.suptitle("Stock PEFT AdaLoRA: inference-mode comparison", fontsize=16)
    figure_path = args.output_dir / "adalora_direct_vs_full_stack.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print("Saved:", figure_path)
    print("Saved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
