"""Compare NBS marginal gains by allocation round across multiple runs.

Each allocation event is mapped to a contiguous round index (1, 2, ...), so
runs with different optimizer-step schedules or early-stopping points can be
compared on the same x-axis.  The diagnostics record the *next candidate* gain
of every LoRA module after an allocation event; this script summarizes those
module-level values for each round.

Example (PowerShell)::

    python analysis/plot_nbs_marginal_gain_rounds.py `
      --diagnostics `
        experiment_downloads/nbs_v6/20260817_073433/nbs_rank_diagnostics.csv `
        experiment_downloads/nbs_v7/20260817_101723/nbs_rank_diagnostics.csv `
        experiment_downloads/nbs_v8/20260817_153454/nbs_rank_diagnostics.csv `
      --labels nbs_v6 nbs_v7 nbs_v8 `
      --output-dir experiment_downloads/nbs_v6_v7_v8_marginal_gain
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_FIELDS = {
    "optimizer_step",
    "event",
    "module_type",
    "rank_delta",
    "next_marginal_utility_gain",
    "next_marginal_gain",
}
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare utility-only and weighted Nash gains by NBS allocation round."
    )
    parser.add_argument(
        "--diagnostics",
        nargs="+",
        required=True,
        type=Path,
        help="One or more nbs_rank_diagnostics.csv files.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional run labels in the same order as --diagnostics.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def optional_float(value: str | None) -> float:
    if value is None or value.strip() == "":
        return math.nan
    return float(value)


def inferred_label(path: Path) -> str:
    return path.parent.parent.name or path.stem


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "nbs"


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def statistics(values: list[float], prefix: str) -> dict[str, float]:
    array = finite(values)
    if not len(array):
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_q25": math.nan,
            f"{prefix}_q75": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
        f"{prefix}_max": float(np.max(array)),
    }


def read_rounds(path: Path, label: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        allocation_rows = [row for row in reader if row["event"] == "allocation"]

    if not allocation_rows:
        raise ValueError(f"No allocation events found in {path}")

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in allocation_rows:
        grouped[int(row["optimizer_step"])].append(row)

    summaries: list[dict[str, Any]] = []
    for round_index, step in enumerate(sorted(grouped), start=1):
        rows = grouped[step]
        utility_values = [optional_float(row["next_marginal_utility_gain"]) for row in rows]
        weighted_values = [optional_float(row["next_marginal_gain"]) for row in rows]
        summary: dict[str, Any] = {
            "label": label,
            "allocation_round": round_index,
            "optimizer_step": step,
            "module_count": len(rows),
            "changed_modules": sum(int(row["rank_delta"]) != 0 for row in rows),
            # A transfer changes donor and recipient ranks, hence division by two.
            "reallocated_rank_units": 0.5 * sum(abs(int(row["rank_delta"])) for row in rows),
        }
        summary.update(statistics(utility_values, "utility_gain"))
        summary.update(statistics(weighted_values, "weighted_nash_gain"))

        for module_type in ("q_proj", "v_proj"):
            selected = [
                optional_float(row["next_marginal_utility_gain"])
                for row in rows
                if row.get("module_type", "") == module_type
            ]
            summary.update(statistics(selected, f"{module_type}_utility_gain"))
        summaries.append(summary)
    return summaries


def configure_gain_axis(axis: Any) -> None:
    axis.set_yscale("symlog", linthresh=1e-8)
    axis.grid(alpha=0.25)


def values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([row[field] for row in rows], dtype=float)


def plot_gain_comparison(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    panels = (
        ("utility_gain_mean", "Mean utility-only candidate gain"),
        ("utility_gain_median", "Median utility-only candidate gain"),
        ("utility_gain_max", "Maximum utility-only candidate gain"),
        ("weighted_nash_gain_mean", "Mean weighted Nash candidate gain"),
    )

    for run_index, run in enumerate(runs):
        rows = run["rounds"]
        x = values(rows, "allocation_round")
        color = COLORS[run_index % len(COLORS)]
        for axis, (field, _) in zip(axes.flat, panels):
            axis.plot(x, values(rows, field), label=run["label"], color=color, linewidth=1.8)

        axes[0, 0].fill_between(
            x,
            values(rows, "utility_gain_q25"),
            values(rows, "utility_gain_q75"),
            color=color,
            alpha=0.12,
        )

    for axis, (_, title) in zip(axes.flat, panels):
        axis.set(title=title, xlabel="NBS allocation round", ylabel="Marginal gain")
        configure_gain_axis(axis)
        axis.legend()

    axes[0, 0].set_title("Mean utility-only candidate gain (shading: module IQR)")
    figure.suptitle("NBS marginal gain by allocation round", fontsize=16)
    output_path = output_dir / "nbs_marginal_gain_by_round.png"
    figure.savefig(output_path, dpi=190)
    plt.close(figure)
    return output_path


def plot_projection_and_churn(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fields = (
        ("q_proj_utility_gain_mean", "q_proj mean utility-only gain"),
        ("v_proj_utility_gain_mean", "v_proj mean utility-only gain"),
    )

    for run_index, run in enumerate(runs):
        rows = run["rounds"]
        x = values(rows, "allocation_round")
        color = COLORS[run_index % len(COLORS)]
        for axis, (field, _) in zip(axes[:2], fields):
            axis.plot(x, values(rows, field), label=run["label"], color=color, linewidth=1.8)
        axes[2].plot(
            x,
            values(rows, "reallocated_rank_units"),
            label=run["label"],
            color=color,
            linewidth=1.8,
        )

    for axis, (_, title) in zip(axes[:2], fields):
        axis.set(title=title, xlabel="NBS allocation round", ylabel="Marginal utility gain")
        configure_gain_axis(axis)
        axis.legend()
    axes[2].set(
        title="Rank movement per allocation round",
        xlabel="NBS allocation round",
        ylabel="Rank units moved",
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.suptitle("Projection-specific gains and allocation activity", fontsize=15)
    output_path = output_dir / "nbs_marginal_gain_projection_and_churn.png"
    figure.savefig(output_path, dpi=190)
    plt.close(figure)
    return output_path


def write_round_csv(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    output_path = output_dir / "nbs_marginal_gain_by_round.csv"
    rows = [row for run in runs for row in run["rounds"]]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_summary(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    summary: dict[str, Any] = {}
    for run in runs:
        rows = run["rounds"]
        mean_gains = finite([row["utility_gain_mean"] for row in rows])
        weighted_gains = finite([row["weighted_nash_gain_mean"] for row in rows])
        summary[run["label"]] = {
            "source": str(run["path"]),
            "allocation_rounds": len(rows),
            "first_optimizer_step": rows[0]["optimizer_step"],
            "last_optimizer_step": rows[-1]["optimizer_step"],
            "initial_mean_utility_gain": float(mean_gains[0]) if len(mean_gains) else None,
            "final_mean_utility_gain": float(mean_gains[-1]) if len(mean_gains) else None,
            "peak_mean_utility_gain": float(np.max(mean_gains)) if len(mean_gains) else None,
            "peak_mean_weighted_nash_gain": (
                float(np.max(weighted_gains)) if len(weighted_gains) else None
            ),
            "total_reallocated_rank_units": float(
                sum(row["reallocated_rank_units"] for row in rows)
            ),
            "rounds_with_rank_changes": sum(row["changed_modules"] > 0 for row in rows),
        }

    output_path = output_dir / "nbs_marginal_gain_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.diagnostics):
        raise ValueError("--labels count must match --diagnostics count")
    labels = args.labels or [inferred_label(path) for path in args.diagnostics]

    runs = []
    for path, label in zip(args.diagnostics, labels):
        runs.append({"path": path, "label": label, "rounds": read_rounds(path, label)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_gain_comparison(runs, args.output_dir),
        plot_projection_and_churn(runs, args.output_dir),
        write_round_csv(runs, args.output_dir),
        write_summary(runs, args.output_dir),
    ]
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
