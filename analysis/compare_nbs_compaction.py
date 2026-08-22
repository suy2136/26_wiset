"""Compare accuracy and latency of original versus compact NBS inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--compact-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="NBS v19")
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_metrics(directory):
    candidates = [
        path for path in directory.glob("*_results.csv")
        if "partial" not in path.name and "per_sample" not in path.name
    ]
    canonical = directory / "results.csv"
    path = canonical if canonical.exists() else (candidates[0] if len(candidates) == 1 else None)
    if path is None:
        raise FileNotFoundError(f"could not uniquely locate results CSV in {directory}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    aggregate = next(
        (row for row in rows if row.get("video") == "-1" and row.get("user") == "-1"),
        None,
    )
    if aggregate is None:
        raise ValueError(f"aggregate -1/-1 row is missing from {path}")
    return {"mae": float(aggregate["mae"]), "rmse": float(aggregate["rmse"]), "path": str(path)}


def latency_metrics(directory):
    path = directory / "latency.json"
    if not path.exists():
        candidates = [path for path in directory.glob("*_latency.json") if "partial" not in path.name]
        if len(candidates) != 1:
            raise FileNotFoundError(f"could not uniquely locate latency JSON in {directory}")
        path = candidates[0]
    data = read_json(path)
    return {
        "mean_ms": float(data["mean_s"]) * 1000.0,
        "median_ms": float(data["median_s"]) * 1000.0,
        "p95_ms": float(data["p95_s"]) * 1000.0,
        "peak_memory_mb": data.get("peak_memory_mb"),
        "measured_calls": int(data["measured_calls"]),
        "path": str(path),
    }


def latency_samples(directory):
    path = directory / "latency_per_sample.csv"
    if not path.exists():
        candidates = list(directory.glob("*_latency_per_sample.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"could not uniquely locate latency samples in {directory}")
        path = candidates[0]
    with path.open(newline="", encoding="utf-8") as handle:
        return [float(row["latency_ms"]) for row in csv.DictReader(handle)]


def relative_change(compact, original):
    return (compact - original) / original * 100.0


def main():
    args = parse_args()
    original = {**aggregate_metrics(args.original_dir), **latency_metrics(args.original_dir)}
    compact = {**aggregate_metrics(args.compact_dir), **latency_metrics(args.compact_dir)}
    metadata = read_json(args.compact_checkpoint / "compaction_metadata.json")
    equivalence = read_json(args.compact_checkpoint / "equivalence_report.json")
    if not equivalence.get("passed"):
        raise RuntimeError("compact checkpoint equivalence report did not pass")

    comparison = {
        "label": args.label,
        "original": original,
        "compact": compact,
        "compact_topology": {
            "physical_rank_total_before": metadata["physical_rank_total_before"],
            "compact_rank_total": metadata["compact_rank_total"],
            "module_count": metadata["module_count"],
            "compression_ratio": (
                metadata["compact_rank_total"] / metadata["physical_rank_total_before"]
            ),
        },
        "equivalence": equivalence,
        "compact_vs_original_percent": {
            "mae": relative_change(compact["mae"], original["mae"]),
            "rmse": relative_change(compact["rmse"], original["rmse"]),
            "mean_latency": relative_change(compact["mean_ms"], original["mean_ms"]),
            "median_latency": relative_change(compact["median_ms"], original["median_ms"]),
            "p95_latency": relative_change(compact["p95_ms"], original["p95_ms"]),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("mode", "mae", "rmse", "mean_ms", "median_ms", "p95_ms", "peak_memory_mb"),
        )
        writer.writeheader()
        writer.writerow({"mode": "original", **{key: original[key] for key in writer.fieldnames if key != "mode"}})
        writer.writerow({"mode": "compact", **{key: compact[key] for key in writer.fieldnames if key != "mode"}})

    labels = ["Original masked\nAdaLoRA", "Compact fixed\nLoRA"]
    colors = ["#4c78a8", "#f58518"]
    original_samples = latency_samples(args.original_dir)
    compact_samples = latency_samples(args.compact_dir)
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    figure.suptitle(f"{args.label}: post-training NBS compaction")

    for axis, key, title, unit in (
        (axes[0, 0], "mae", "Aggregate MAE", "°"),
        (axes[0, 1], "rmse", "Aggregate RMSE", "°"),
        (axes[0, 2], "mean_ms", "Mean inference latency", " ms"),
    ):
        bars = axis.bar(labels, [original[key], compact[key]], color=colors)
        axis.bar_label(bars, labels=[f"{bar.get_height():.3f}{unit}" for bar in bars], padding=3)
        axis.set_title(f"{title} (lower is better)")
        axis.grid(axis="y", alpha=0.25)

    positions = range(3)
    width = 0.36
    for index, (row, label, color) in enumerate(zip((original, compact), labels, colors)):
        offset = (index - 0.5) * width
        axes[1, 0].bar(
            [position + offset for position in positions],
            [row["mean_ms"], row["median_ms"], row["p95_ms"]],
            width=width, label=label.replace("\n", " "), color=color,
        )
    axes[1, 0].set_xticks(list(positions), ["Mean", "Median", "P95"])
    axes[1, 0].set(title="Latency distribution summary", ylabel="ms / sample")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.25)

    for samples, label, color in zip((original_samples, compact_samples), labels, colors):
        ordered = sorted(samples)
        probability = [(index + 1) / len(ordered) for index in range(len(ordered))]
        axes[1, 1].plot(ordered, probability, label=label.replace("\n", " "), color=color)
    axes[1, 1].set(title="Per-sample latency ECDF", xlabel="Latency (ms)", ylabel="Cumulative probability")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.25)

    before = metadata["physical_rank_total_before"]
    after = metadata["compact_rank_total"]
    bars = axes[1, 2].bar(["Physical slots\nbefore", "Active slots\nafter"], [before, after], color=colors)
    axes[1, 2].bar_label(bars, padding=3)
    axes[1, 2].set(title="Adapter rank storage/compute width", ylabel="Total rank slots")
    axes[1, 2].grid(axis="y", alpha=0.25)

    path = args.output_dir / "nbs_original_vs_compact.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print("Saved:", path)
    print("Saved:", args.output_dir / "comparison.json")


if __name__ == "__main__":
    main()
