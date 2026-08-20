"""Compare accuracy, latency, and inference behavior for selector/speculative ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = (
    ("1_nbs_selector", "NBS +\nSelector"),
    ("2_nbs_speculative", "NBS +\nSpeculative"),
    ("3_nbs_full_stack", "NBS +\nSelector + Spec."),
    ("4_uniform_b736_full_stack", "Uniform-b736 +\nSelector + Spec."),
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_case(run_dir: Path, directory: str, label: str) -> dict:
    case_dir = run_dir / directory
    summaries = list((case_dir / "figures").glob("summary.json"))
    if len(summaries) != 1:
        raise FileNotFoundError(f"Expected one summary.json under {case_dir / 'figures'}")
    performance = read_json(summaries[0])
    latency = read_json(case_dir / "latency.json")
    trace = read_json(case_dir / "inference_trace.json")
    return {
        "case": directory,
        "label": label,
        "mae": performance.get("aggregate_mae"),
        "rmse": performance.get("aggregate_rmse"),
        "latency_mean_ms": latency.get("mean_s") * 1000.0,
        "latency_median_ms": latency.get("median_s") * 1000.0,
        "latency_p95_ms": latency.get("p95_s") * 1000.0,
        "peak_memory_mb": latency.get("peak_memory_mb"),
        "mean_initial_tokens": trace.get("mean_initial_token_count"),
        "mean_selected_tokens": trace.get("mean_selected_token_count"),
        "token_reduction_percent": trace.get("mean_token_reduction_percent"),
        "mean_target_forward_count": trace.get("mean_target_forward_count"),
        "draft_acceptance_rate": trace.get("draft_acceptance_rate"),
    }


def bar(axis, labels, values, title, ylabel, color):
    positions = range(len(labels))
    bars = axis.bar(positions, values, color=color)
    axis.set_xticks(list(positions), labels)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    for item, value in zip(bars, values):
        if value is not None:
            axis.text(
                item.get_x() + item.get_width() / 2,
                item.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [load_case(args.run_dir, directory, label) for directory, label in CASES]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    fig, axes = plt.subplots(2, 4, figsize=(23, 10), constrained_layout=True)
    bar(axes[0, 0], labels, [row["mae"] for row in rows], "MAE", "Degrees", "#2a9d8f")
    bar(axes[0, 1], labels, [row["rmse"] for row in rows], "RMSE", "Degrees", "#6c6fb3")
    bar(
        axes[0, 2], labels, [row["latency_mean_ms"] for row in rows],
        "Mean inference latency", "ms/sample", "#e76f51",
    )
    bar(
        axes[0, 3], labels, [row["latency_p95_ms"] for row in rows],
        "p95 inference latency", "ms/sample", "#f4a261",
    )
    bar(
        axes[1, 0], labels,
        [row["token_reduction_percent"] or 0.0 for row in rows],
        "Selector token reduction", "%", "#457b9d",
    )
    bar(
        axes[1, 1], labels,
        [row["mean_target_forward_count"] or 0.0 for row in rows],
        "Target-model forward count", "Calls/sample", "#7b2cbf",
    )
    bar(
        axes[1, 2], labels,
        [
            (row["draft_acceptance_rate"] * 100.0)
            if row["draft_acceptance_rate"] is not None else 0.0
            for row in rows
        ],
        "Speculative draft acceptance", "%", "#8ab17d",
    )
    bar(
        axes[1, 3], labels, [row["peak_memory_mb"] for row in rows],
        "Peak allocated GPU memory", "MiB", "#bc6c25",
    )
    fig.suptitle("Selector / speculative decoding ablation", fontsize=16)
    figure_path = args.output_dir / "selector_speculative_ablation.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print("Saved:", figure_path)
    print("Saved:", args.output_dir / "comparison.csv")
    print("Saved:", args.output_dir / "comparison.json")


if __name__ == "__main__":
    main()
