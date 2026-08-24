"""Compare MAE, RMSE, and inference latency across NBS experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-role", default="best_ar")
    args = parser.parse_args()
    if len(args.runs) != len(args.labels):
        parser.error("--runs and --labels must have the same length")
    return args


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(run_dir: Path, label: str, checkpoint_role: str) -> dict:
    report_path = run_dir / "checkpoint_nbs_report" / "report_summary.json"
    if report_path.is_file():
        report = read_json(report_path)
        metrics = report.get("checkpoint_metrics", {}).get(checkpoint_role)
        if metrics is not None:
            latency = metrics.get("latency") or {}
            return {
                "label": label,
                "checkpoint_role": checkpoint_role,
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "latency_mean_ms": float(latency["mean_s"]) * 1000.0,
                "latency_p95_ms": float(latency["p95_s"]) * 1000.0,
                "final_total_rank": report.get("final_total_rank"),
                "final_mean_rank": report.get("final_rank_mean"),
                "run_dir": str(run_dir),
            }

    evaluation_dir = run_dir / "evaluations" / checkpoint_role
    results_path = evaluation_dir / "results.csv"
    latency_path = evaluation_dir / "latency.json"
    if not results_path.is_file() or not latency_path.is_file():
        raise FileNotFoundError(
            f"Missing {checkpoint_role} results/latency under {run_dir}"
        )
    with results_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows in {results_path}")
    latency = read_json(latency_path)
    return {
        "label": label,
        "checkpoint_role": checkpoint_role,
        "mae": sum(float(row["mae"]) for row in rows) / len(rows),
        "rmse": sum(float(row["rmse"]) for row in rows) / len(rows),
        "latency_mean_ms": float(latency["mean_s"]) * 1000.0,
        "latency_p95_ms": float(latency["p95_s"]) * 1000.0,
        "final_total_rank": None,
        "final_mean_rank": None,
        "run_dir": str(run_dir),
    }


def annotate(axis, bars, values, decimals=2) -> None:
    top = max(values) if values else 1.0
    offset = max(top * 0.02, 0.02)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    args = parse_args()
    rows = [
        load_metrics(run_dir, label, args.checkpoint_role)
        for run_dir, label in zip(args.runs, args.labels)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "nbs_performance_comparison.csv"
    json_path = args.output_dir / "nbs_performance_comparison.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    labels = [row["label"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    settings = (
        ("mae", "MAE (lower is better)", "Degrees", "#2a9d8f"),
        ("rmse", "RMSE (lower is better)", "Degrees", "#6c6fb3"),
    )
    for axis, (key, title, ylabel, color) in zip(axes[:2], settings):
        values = [row[key] for row in rows]
        bars = axis.bar(labels, values, color=color)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=15)
        annotate(axis, bars, values)

    positions = list(range(len(labels)))
    width = 0.36
    means = [row["latency_mean_ms"] for row in rows]
    p95s = [row["latency_p95_ms"] for row in rows]
    mean_bars = axes[2].bar(
        [value - width / 2 for value in positions], means, width,
        label="Mean", color="#e76f51",
    )
    p95_bars = axes[2].bar(
        [value + width / 2 for value in positions], p95s, width,
        label="p95", color="#f4a261",
    )
    axes[2].set_xticks(positions, labels, rotation=15)
    axes[2].set(
        title="Inference latency (lower is better)", ylabel="ms / sample"
    )
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()
    annotate(axes[2], mean_bars, means, decimals=1)
    annotate(axes[2], p95_bars, p95s, decimals=1)

    figure.suptitle(
        f"NBS performance comparison — checkpoint: {args.checkpoint_role}",
        fontsize=14,
    )
    figure_path = args.output_dir / "nbs_mae_rmse_latency_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    for row in rows:
        print(
            f'{row["label"]}: MAE={row["mae"]:.4f}, '
            f'RMSE={row["rmse"]:.4f}, mean latency={row["latency_mean_ms"]:.2f} ms, '
            f'p95={row["latency_p95_ms"]:.2f} ms'
        )
    for path in (figure_path, csv_path, json_path):
        print("Saved:", path)


if __name__ == "__main__":
    main()
