"""Plot rule-based cross-dataset and all-method Wu2017 unseen performance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RULE_CASES = (
    ("1_regression_Jin2022", "Linear Regression", "Jin2022"),
    ("2_velocity_Jin2022", "Velocity", "Jin2022"),
    ("3_regression_Wu2017", "Linear Regression", "Wu2017"),
    ("4_velocity_Wu2017", "Velocity", "Wu2017"),
)
MODEL_CASES = (
    ("1_nbs_v12_repeat_direct", "NBS v12 direct"),
    ("2_nbs_v12_repeat_selector", "NBS v12 + Selector"),
    ("3_nbs_v12_repeat_speculative", "NBS v12 + Speculative"),
    ("4_nbs_v12_repeat_full_stack", "NBS v12 full stack"),
    ("5_uniform_b736_direct", "Uniform-b736 direct"),
    ("6_adalora_direct", "AdaLoRA direct"),
    ("7_uniform_b736_full_stack", "Uniform-b736 full stack"),
    ("8_adalora_full_stack", "AdaLoRA full stack"),
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_metrics(path: Path) -> tuple[float, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    aggregate = next(
        (row for row in rows if int(float(row["video"])) == -1 and int(float(row["user"])) == -1),
        None,
    )
    if aggregate is None:
        raise ValueError(f"Aggregate video=-1,user=-1 row missing in {path}")
    return float(aggregate["mae"]), float(aggregate["rmse"])


def load_case(case_dir: Path, label: str, dataset: str, family: str) -> dict:
    status = read_json(case_dir / "status.json")
    if status.get("status") != "complete":
        raise RuntimeError(f"Case is not complete: {case_dir} ({status})")
    mae, rmse = aggregate_metrics(case_dir / "results.csv")
    latency = read_json(case_dir / "latency.json")
    if latency.get("mean_s") is None:
        raise ValueError(f"No measured latency samples in {case_dir / 'latency.json'}")
    return {
        "family": family,
        "label": label,
        "dataset": dataset,
        "mae": mae,
        "rmse": rmse,
        "latency_mean_ms": float(latency["mean_s"]) * 1000.0,
        "latency_median_ms": float(latency["median_s"]) * 1000.0,
        "latency_p95_ms": float(latency["p95_s"]) * 1000.0,
        "measured_calls": int(latency["measured_calls"]),
        "case_dir": str(case_dir),
    }


def save_rows(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "wu2017_unseen_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "wu2017_unseen_comparison.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def plot_rule_cross_dataset(rows: list[dict], output_dir: Path) -> Path:
    algorithms = ("Linear Regression", "Velocity")
    datasets = ("Jin2022", "Wu2017")
    metrics = (
        ("mae", "MAE", "Degrees"),
        ("rmse", "RMSE", "Degrees"),
        ("latency_mean_ms", "Mean inference latency", "ms / sample"),
    )
    lookup = {(row["label"], row["dataset"]): row for row in rows}
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    x = np.arange(len(algorithms))
    width = 0.36
    colors = ("#4c78a8", "#f58518")
    for axis, (key, title, ylabel) in zip(axes, metrics):
        for index, (dataset, color) in enumerate(zip(datasets, colors)):
            values = [lookup[(algorithm, dataset)][key] for algorithm in algorithms]
            bars = axis.bar(x + (index - 0.5) * width, values, width, label=dataset, color=color)
            axis.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
        axis.set_xticks(x, algorithms)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].legend()
    figure.suptitle("Rule-based viewport prediction: seen vs unseen dataset", fontsize=15)
    path = output_dir / "rule_based_jin2022_vs_wu2017.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_wu_all_methods(rows: list[dict], output_dir: Path) -> Path:
    metrics = (
        ("mae", "MAE (lower is better)", "Degrees"),
        ("rmse", "RMSE (lower is better)", "Degrees"),
        ("latency_mean_ms", "Mean latency (lower is better)", "ms / sample"),
    )
    labels = [row["label"] for row in rows]
    positions = np.arange(len(rows))
    colors = ["#9c755f" if row["family"] == "rule_based" else "#4c78a8" for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(21, 9), sharey=True, constrained_layout=True)
    for axis, (key, title, xlabel) in zip(axes, metrics):
        values = [row[key] for row in rows]
        bars = axis.barh(positions, values, color=colors)
        axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.margins(x=0.16)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    figure.suptitle(
        "Wu2017 unseen evaluation — rule-based baselines and Jin2022-trained models",
        fontsize=15,
    )
    path = output_dir / "wu2017_all_methods_performance.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-run-dir", type=Path, required=True)
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rule_rows = [
        load_case(args.rule_run_dir / directory, label, dataset, "rule_based")
        for directory, label, dataset in RULE_CASES
    ]
    model_rows = [
        load_case(args.model_run_dir / directory, label, "Wu2017", "checkpoint_model")
        for directory, label in MODEL_CASES
    ]
    wu_rows = [row for row in rule_rows if row["dataset"] == "Wu2017"] + model_rows

    rule_figure = plot_rule_cross_dataset(rule_rows, args.output_dir)
    all_figure = plot_wu_all_methods(wu_rows, args.output_dir)
    csv_path, json_path = save_rows(rule_rows + model_rows, args.output_dir)
    print("Saved:", rule_figure)
    print("Saved:", all_figure)
    print("Saved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
