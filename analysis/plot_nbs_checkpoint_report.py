"""Create a checkpoint-performance and seven-statistic NBS report.

The input is one downloaded experiment run directory produced by
``scripts/run_netllm_experiment.sh``.  New runs contain one evaluation folder
per logical checkpoint role; legacy runs with only top-level results are also
accepted as a single ``best_ar`` evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROLE_ORDER = ("best_ar", "best_post_nbs", "final_nbs")
ROLE_LABELS = {
    "best_ar": "Best AR",
    "best_post_nbs": "Best post-NBS",
    "final_nbs": "Final NBS",
}
STATISTICS = (
    ("rank", "Allocated rank", False),
    ("sensitivity", "EMA sensitivity", True),
    ("alpha", "Bargaining weight α", True),
    ("spectral_energy_total", "Spectral energy", True),
    ("utility", "Normalized utility U(r)", False),
    ("next_marginal_utility_gain", "Marginal utility log-gain", True),
    ("next_marginal_gain", "Marginal Nash gain", True),
)
MODULE_COLORS = {"q_proj": "#2878b5", "v_proj": "#f28e2b", "all": "#666666"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize NBS checkpoint performance and allocator statistics."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="NBS v6")
    return parser.parse_args()


def optional_float(value: str | None) -> float:
    if value is None or value.strip() == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_result_csv(path: Path) -> tuple[dict[str, float], list[dict[str, float]]]:
    aggregate = None
    samples: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                "video": int(float(row["video"])),
                "user": int(float(row["user"])),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
            }
            if parsed["video"] == -1 and parsed["user"] == -1:
                aggregate = {"mae": parsed["mae"], "rmse": parsed["rmse"]}
            else:
                samples.append(parsed)
    if aggregate is None:
        finite = [
            row for row in samples
            if math.isfinite(row["mae"]) and math.isfinite(row["rmse"])
        ]
        if not finite:
            raise ValueError(f"No finite metrics found in {path}")
        aggregate = {
            "mae": float(np.mean([row["mae"] for row in finite])),
            "rmse": float(np.mean([row["rmse"] for row in finite])),
        }
    return aggregate, samples


def read_latency_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def discover_checkpoint_results(run_dir: Path) -> list[dict[str, Any]]:
    discovered = []
    for role in ROLE_ORDER:
        role_dir = run_dir / "evaluations" / role
        result_path = role_dir / "results.csv"
        if not result_path.exists():
            continue
        aggregate, samples = read_result_csv(result_path)
        latency = read_latency_json(role_dir / "latency.json")
        reused_path = role_dir / "evaluation_reused.json"
        reused = (
            json.loads(reused_path.read_text(encoding="utf-8"))
            if reused_path.exists() else None
        )
        discovered.append({
            "role": role,
            "label": ROLE_LABELS[role],
            "result_path": str(result_path),
            "aggregate": aggregate,
            "samples": samples,
            "latency": latency,
            "reused_evaluation": reused,
        })
    if not discovered and (run_dir / "results.csv").exists():
        aggregate, samples = read_result_csv(run_dir / "results.csv")
        latency = read_latency_json(run_dir / "latency.json")
        discovered.append({
            "role": "best_ar",
            "label": ROLE_LABELS["best_ar"],
            "result_path": str(run_dir / "results.csv"),
            "aggregate": aggregate,
            "samples": samples,
            "latency": latency,
            "reused_evaluation": None,
        })
    if not discovered:
        raise FileNotFoundError(
            f"No checkpoint evaluation results found under {run_dir}"
        )
    return discovered


def module_coordinates(row: dict[str, str]) -> tuple[int, int, str]:
    layer_text = row.get("transformer_layer_index", "").strip()
    if layer_text:
        layer_index = int(float(layer_text))
    else:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", row["layer_name"])
        layer_index = int(match.group(1)) if match else 10**9
    module_type = row.get("module_type", "").strip() or row["layer_name"].rsplit(".", 1)[-1]
    return layer_index, {"q_proj": 0, "v_proj": 1}.get(module_type, 99), module_type


def read_diagnostics(path: Path) -> tuple[
    list[dict[str, str]], OrderedDict[tuple[int, str], list[dict[str, str]]]
]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No diagnostic rows found in {path}")
    required = {"optimizer_step", "event", "layer_name", "rank"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Diagnostics missing fields: {sorted(missing)}")
    events: OrderedDict[tuple[int, str], list[dict[str, str]]] = OrderedDict()
    for row in rows:
        events.setdefault((int(row["optimizer_step"]), row["event"]), []).append(row)
    return rows, events


def preferred_events(
    events: OrderedDict[tuple[int, str], list[dict[str, str]]]
) -> list[tuple[tuple[int, str], list[dict[str, str]]]]:
    allocation = [(key, group) for key, group in events.items() if key[1] == "allocation"]
    return allocation or list(events.items())


def event_statistic(
    selected_events: list[tuple[tuple[int, str], list[dict[str, str]]]],
    field: str,
    module_type: str,
) -> tuple[list[int], list[float]]:
    steps, means = [], []
    for (step, _), group in selected_events:
        values = []
        for row in group:
            if module_type != "all" and module_coordinates(row)[2] != module_type:
                continue
            value = optional_float(row.get(field))
            if math.isfinite(value):
                values.append(value)
        if values:
            steps.append(step)
            means.append(float(np.mean(values)))
    return steps, means


def plot_checkpoint_performance(
    checkpoints: list[dict[str, Any]], label: str, output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(21, 5.2), constrained_layout=True)
    figure.suptitle(f"{label} — checkpoint performance comparison", fontsize=15)
    labels = [item["label"] for item in checkpoints]
    x = np.arange(len(checkpoints))
    width = 0.36
    mae = [item["aggregate"]["mae"] for item in checkpoints]
    rmse = [item["aggregate"]["rmse"] for item in checkpoints]
    axes[0].bar(x - width / 2, mae, width, label="MAE", color="#2a9d8f")
    axes[0].bar(x + width / 2, rmse, width, label="RMSE", color="#6c5ce7")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set(title="Aggregate metrics", ylabel="Error (lower is better)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for axis, metric, color in (
        (axes[1], "mae", "#7cb342"),
        (axes[2], "rmse", "#f0b323"),
    ):
        distributions = [
            [row[metric] for row in item["samples"] if math.isfinite(row[metric])]
            for item in checkpoints
        ]
        boxplot_kwargs = {
            "showfliers": True,
            "patch_artist": True,
        }
        try:
            # Matplotlib 3.9+ renamed ``labels`` to ``tick_labels`` and
            # recent releases no longer accept the old keyword.
            boxes = axis.boxplot(
                distributions, tick_labels=labels, **boxplot_kwargs
            )
        except TypeError as error:
            if "tick_labels" not in str(error):
                raise
            boxes = axis.boxplot(
                distributions, labels=labels, **boxplot_kwargs
            )
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axis.set(title=f"Per video-user {metric.upper()}", ylabel=metric.upper())
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    latency_available = all(
        item["latency"] and item["latency"].get("mean_s") is not None
        for item in checkpoints
    )
    if latency_available:
        mean_ms = [item["latency"]["mean_s"] * 1000.0 for item in checkpoints]
        p95_ms = [item["latency"]["p95_s"] * 1000.0 for item in checkpoints]
        axes[3].bar(x - width / 2, mean_ms, width, label="Mean", color="#4c78a8")
        axes[3].bar(x + width / 2, p95_ms, width, label="P95", color="#e45756")
        axes[3].set_xticks(x, labels, rotation=15, ha="right")
        axes[3].set(title="Inference latency", ylabel="Milliseconds (lower is better)")
        axes[3].legend()
        axes[3].grid(axis="y", alpha=0.25)
    else:
        axes[3].text(0.5, 0.5, "Latency not recorded", ha="center", va="center")
        axes[3].set_title("Inference latency")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_seven_statistics(
    selected_events: list[tuple[tuple[int, str], list[dict[str, str]]]],
    label: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(17, 18), constrained_layout=True)
    figure.suptitle(f"{label} — seven NBS allocator statistics", fontsize=16)
    flat_axes = axes.reshape(-1)
    for axis, (field, title, prefer_log) in zip(flat_axes, STATISTICS):
        plotted_positive = []
        for module_type in ("q_proj", "v_proj"):
            steps, means = event_statistic(selected_events, field, module_type)
            if not steps:
                continue
            axis.plot(
                steps, means, marker="o", markersize=3, linewidth=1.5,
                label=module_type, color=MODULE_COLORS[module_type],
            )
            plotted_positive.extend(value for value in means if value > 0)
        if prefer_log and plotted_positive:
            axis.set_yscale("log")
        axis.set(title=title, xlabel="Optimizer step", ylabel="Mean across modules")
        axis.grid(alpha=0.25)
        axis.legend()

    churn_axis = flat_axes[-1]
    steps, moved = [], []
    for (step, _), group in selected_events:
        deltas = [abs(int(float(row.get("rank_delta", "0") or 0))) for row in group]
        steps.append(step)
        moved.append(0.5 * sum(deltas))
    churn_axis.bar(steps, moved, color="#555555", width=max(1, len(steps) / 3))
    churn_axis.set(
        title="Supplement: rank units moved", xlabel="Optimizer step", ylabel="Moved units"
    )
    churn_axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_rank_trajectory(
    rows: list[dict[str, str]],
    selected_events: list[tuple[tuple[int, str], list[dict[str, str]]]],
    label: str,
    output_path: Path,
) -> None:
    module_meta = {row["layer_name"]: module_coordinates(row) for row in rows}
    modules = sorted(module_meta, key=lambda name: (*module_meta[name], name))
    event_keys = [key for key, _ in selected_events]
    lookup = {
        (key, row["layer_name"]): int(float(row["rank"]))
        for key, group in selected_events for row in group
    }
    matrix = np.asarray([
        [lookup.get((key, module), np.nan) for key in event_keys]
        for module in modules
    ])
    figure, axis = plt.subplots(figsize=(max(13, len(event_keys) * 0.35), 13), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    tick_count = min(12, len(event_keys))
    ticks = sorted(set(np.linspace(0, len(event_keys) - 1, tick_count, dtype=int)))
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [f"{event_keys[index][0]}\n{event_keys[index][1]}" for index in ticks],
        rotation=45, ha="right", fontsize=8,
    )
    axis.set_yticks(range(len(modules)))
    axis.set_yticklabels([
        f"L{module_meta[name][0]} {module_meta[name][2]}" for name in modules
    ], fontsize=7)
    axis.set(
        title=f"{label} — NBS rank trajectory",
        xlabel="Optimizer step / event", ylabel="LoRA module",
    )
    figure.colorbar(image, ax=axis, label="Allocated rank")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_checkpoint_csv(checkpoints: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "checkpoint_role", "mae", "rmse", "latency_mean_ms",
                "latency_median_ms", "latency_p95_ms", "reused_from_role",
                "result_path",
            ),
        )
        writer.writeheader()
        for item in checkpoints:
            reused = item["reused_evaluation"] or {}
            latency = item["latency"] or {}
            writer.writerow({
                "checkpoint_role": item["role"],
                "mae": item["aggregate"]["mae"],
                "rmse": item["aggregate"]["rmse"],
                "latency_mean_ms": (
                    latency["mean_s"] * 1000.0 if latency.get("mean_s") is not None else ""
                ),
                "latency_median_ms": (
                    latency["median_s"] * 1000.0 if latency.get("median_s") is not None else ""
                ),
                "latency_p95_ms": (
                    latency["p95_s"] * 1000.0 if latency.get("p95_s") is not None else ""
                ),
                "reused_from_role": reused.get("reused_from_role", ""),
                "result_path": item["result_path"],
            })


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "checkpoint_nbs_report").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = run_dir / "nbs_rank_diagnostics.csv"
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Diagnostics not found: {diagnostics_path}")

    checkpoints = discover_checkpoint_results(run_dir)
    rows, events = read_diagnostics(diagnostics_path)
    selected_events = preferred_events(events)

    plot_checkpoint_performance(
        checkpoints, args.label, output_dir / "checkpoint_performance.png"
    )
    plot_seven_statistics(
        selected_events, args.label, output_dir / "nbs_seven_statistics.png"
    )
    plot_rank_trajectory(
        rows, selected_events, args.label, output_dir / "nbs_rank_trajectory.png"
    )
    write_checkpoint_csv(checkpoints, output_dir / "checkpoint_metrics.csv")

    final_group = selected_events[-1][1]
    final_ranks = [int(float(row["rank"])) for row in final_group]
    allocation_event_count = sum(key[1] == "allocation" for key in events)
    summary = {
        "label": args.label,
        "run_dir": str(run_dir),
        "diagnostics": str(diagnostics_path),
        "checkpoint_metrics": {
            item["role"]: {
                **item["aggregate"],
                "latency": item["latency"],
                "reused_evaluation": item["reused_evaluation"],
            }
            for item in checkpoints
        },
        "nbs_statistics": [field for field, _, _ in STATISTICS],
        "allocation_event_count": allocation_event_count,
        "visualized_event_count": len(selected_events),
        "final_rank_min": min(final_ranks),
        "final_rank_max": max(final_ranks),
        "final_rank_mean": float(np.mean(final_ranks)),
        "final_total_rank": int(sum(final_ranks)),
    }
    (output_dir / "report_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Checkpoint comparison saved at {output_dir / 'checkpoint_performance.png'}")
    print(f"Seven NBS statistics saved at {output_dir / 'nbs_seven_statistics.png'}")
    print(f"Rank trajectory saved at {output_dir / 'nbs_rank_trajectory.png'}")
    print(f"Report summary saved at {output_dir / 'report_summary.json'}")


if __name__ == "__main__":
    main()
