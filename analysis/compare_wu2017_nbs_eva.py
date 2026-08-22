"""Compare unseen-Wu2017 accuracy and latency for NBS v19 and EVA."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbs-dir", type=Path, required=True)
    parser.add_argument("--eva-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_case(directory):
    with (directory / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    aggregate = next(
        row for row in rows
        if row.get("video") == "-1" and row.get("user") == "-1"
    )
    latency = json.loads((directory / "latency.json").read_text(encoding="utf-8"))
    return {
        "mae": float(aggregate["mae"]),
        "rmse": float(aggregate["rmse"]),
        "mean_latency_ms": float(latency["mean_s"]) * 1000.0,
        "median_latency_ms": float(latency["median_s"]) * 1000.0,
        "p95_latency_ms": float(latency["p95_s"]) * 1000.0,
    }


def main():
    args = parse_args()
    rows = [load_case(args.nbs_dir), load_case(args.eva_dir)]
    labels = ["NBS v19", "EVA"]
    colors = ["#4c78a8", "#f58518"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {label: row for label, row in zip(labels, rows)}
    (args.output_dir / "wu2017_nbs_v19_vs_eva.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "wu2017_nbs_v19_vs_eva.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", *rows[0].keys()))
        writer.writeheader()
        for label, row in zip(labels, rows):
            writer.writerow({"model": label, **row})

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, key, title, suffix in (
        (axes[0], "mae", "Unseen Wu2017 MAE", "°"),
        (axes[1], "rmse", "Unseen Wu2017 RMSE", "°"),
        (axes[2], "mean_latency_ms", "Mean inference latency", " ms"),
    ):
        bars = axis.bar(labels, [row[key] for row in rows], color=colors)
        axis.bar_label(
            bars, labels=[f"{bar.get_height():.2f}{suffix}" for bar in bars], padding=3
        )
        axis.set_title(f"{title}\n(lower is better)")
        axis.grid(axis="y", alpha=0.25)
    path = args.output_dir / "wu2017_nbs_v19_vs_eva.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print("Saved:", path)


if __name__ == "__main__":
    main()
