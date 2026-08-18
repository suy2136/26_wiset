"""Relate NBS bargaining-weight dispersion, rank dispersion, and final MAE.

The two requested comparisons use one consistent training endpoint:

1. Var(alpha_l) versus Var(r_l) at the last actual allocation event.
2. Mean absolute rank deviation versus final_nbs autoregressive MAE.

The first comparison intentionally uses allocation-time alpha rather than a
cooldown-end alpha that could no longer have affected ranks.  The second uses
the final model ranks and the matching final_nbs evaluation MAE.
"""

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
import numpy as np


COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#9467bd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Var(alpha) vs Var(rank) and rank heterogeneity vs final MAE."
    )
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def select_final_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], str, int]:
    candidates = [row for row in rows if row.get("event") == "final_nbs_model"]
    source_event = "final_nbs_model"
    if not candidates:
        candidates = [row for row in rows if row.get("event") == "allocation"]
        source_event = "latest_allocation"
    if not candidates:
        raise ValueError("Diagnostics contain neither final_nbs_model nor allocation rows")
    step = max(int(float(row["optimizer_step"])) for row in candidates)
    return (
        [row for row in candidates if int(float(row["optimizer_step"])) == step],
        source_event,
        step,
    )


def select_last_allocation_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    candidates = [row for row in rows if row.get("event") == "allocation"]
    if not candidates:
        raise ValueError("Diagnostics contain no allocation rows")
    step = max(int(float(row["optimizer_step"])) for row in candidates)
    return (
        [row for row in candidates if int(float(row["optimizer_step"])) == step],
        step,
    )


def load_run(run_dir: Path, label: str) -> dict[str, Any]:
    diagnostics_path = run_dir / "nbs_rank_diagnostics.csv"
    report_path = run_dir / "checkpoint_nbs_report" / "report_summary.json"
    if not diagnostics_path.exists():
        raise FileNotFoundError(diagnostics_path)
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    diagnostics = read_csv(diagnostics_path)
    rows, source_event, step = select_final_rows(diagnostics)
    allocation_rows, allocation_step = select_last_allocation_rows(diagnostics)
    # Deduplicate by layer so repeated final rows cannot bias the statistics.
    by_layer = {row["layer_name"]: row for row in rows}
    ranks = np.asarray([float(row["rank"]) for row in by_layer.values()])
    allocation_by_layer = {row["layer_name"]: row for row in allocation_rows}
    allocation_ranks = np.asarray([
        float(row["rank"]) for row in allocation_by_layer.values()
    ])
    alphas = np.asarray([
        float(row["alpha"]) for row in allocation_by_layer.values()
    ])
    if ranks.size == 0 or alphas.size != allocation_ranks.size:
        raise ValueError(f"Incomplete rank/alpha diagnostics in {diagnostics_path}")
    if not np.all(np.isfinite(alphas)):
        raise ValueError(f"Non-finite alpha values in {diagnostics_path}")

    mean_rank = float(np.mean(ranks))
    rank_variance = float(np.var(ranks))
    rank_mad = float(np.mean(np.abs(ranks - mean_rank)))
    report = read_json(report_path)
    final_mae = float(report["checkpoint_metrics"]["final_nbs"]["mae"])
    return {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "source_event": source_event,
        "optimizer_step": step,
        "allocation_optimizer_step": allocation_step,
        "module_count": int(ranks.size),
        "total_rank": int(np.sum(ranks)),
        "mean_rank": mean_rank,
        "rank_variance": rank_variance,
        "allocation_rank_variance": float(np.var(allocation_ranks)),
        "rank_mad": rank_mad,
        "normalized_rank_mad": rank_mad / mean_rank if mean_rank else 0.0,
        "alpha_mean": float(np.mean(alphas)),
        "alpha_variance": float(np.var(alphas)),
        "alpha_std": float(np.std(alphas)),
        "final_nbs_mae": final_mae,
    }


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def add_points(axis: Any, x: np.ndarray, y: np.ndarray, runs: list[dict[str, Any]]) -> None:
    x_span = float(np.ptp(x)) or 1.0
    y_span = float(np.ptp(y)) or 1.0
    for index, run in enumerate(runs):
        axis.scatter(
            x[index], y[index], s=85, color=COLORS[index % len(COLORS)],
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        near_right = x[index] >= float(np.max(x)) - 0.1 * x_span
        near_top = y[index] >= float(np.max(y)) - 0.1 * y_span
        axis.annotate(
            run["label"], (x[index], y[index]),
            xytext=(-7 if near_right else 7, -7 if near_top else 6),
            textcoords="offset points",
            ha="right" if near_right else "left",
            va="top" if near_top else "bottom",
            fontsize=9,
        )


def set_padded_limits(axis: Any, x: np.ndarray, y: np.ndarray) -> None:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_padding = (x_max - x_min) * 0.08 or max(abs(x_min) * 0.08, 1e-12)
    y_padding = (y_max - y_min) * 0.08 or max(abs(y_min) * 0.08, 1e-12)
    axis.set_xlim(x_min - x_padding, x_max + x_padding)
    axis.set_ylim(y_min - y_padding, y_max + y_padding)


def add_trend(axis: Any, x: np.ndarray, y: np.ndarray) -> None:
    if x.size < 2 or np.allclose(x, x[0]):
        return
    coefficient = np.polyfit(x, y, 1)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    axis.plot(x_line, np.polyval(coefficient, x_line), color="#777777", linestyle="--", linewidth=1.2)


def plot(runs: list[dict[str, Any]], output_path: Path) -> dict[str, float | None]:
    alpha_variance = np.asarray([run["alpha_variance"] for run in runs])
    rank_variance = np.asarray([run["allocation_rank_variance"] for run in runs])
    rank_mad = np.asarray([run["rank_mad"] for run in runs])
    final_mae = np.asarray([run["final_nbs_mae"] for run in runs])
    correlations = {
        "pearson_alpha_variance_vs_rank_variance": correlation(alpha_variance, rank_variance),
        "pearson_rank_mad_vs_final_nbs_mae": correlation(rank_mad, final_mae),
    }

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    figure.suptitle("NBS heterogeneity relationships at the final model", fontsize=15)

    add_points(axes[0], alpha_variance, rank_variance, runs)
    add_trend(axes[0], alpha_variance, rank_variance)
    set_padded_limits(axes[0], alpha_variance, rank_variance)
    corr = correlations["pearson_alpha_variance_vs_rank_variance"]
    axes[0].set(
        title=f"Bargaining-weight dispersion vs rank dispersion (Pearson r={corr:.3f})",
        xlabel="Var(alpha_l)",
        ylabel="Var(r_l) (rank²)",
    )
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].grid(alpha=0.25)

    add_points(axes[1], rank_mad, final_mae, runs)
    add_trend(axes[1], rank_mad, final_mae)
    set_padded_limits(axes[1], rank_mad, final_mae)
    corr = correlations["pearson_rank_mad_vs_final_nbs_mae"]
    axes[1].set(
        title=f"Rank heterogeneity vs final prediction error (Pearson r={corr:.3f})",
        xlabel="Mean |r_l - R/L| (rank)",
        ylabel="Final NBS MAE (degrees; lower is better)",
    )
    axes[1].grid(alpha=0.25)

    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return correlations


def write_outputs(
    runs: list[dict[str, Any]], correlations: dict[str, float | None], output_dir: Path
) -> tuple[Path, Path]:
    csv_path = output_dir / "nbs_heterogeneity_relations.csv"
    json_path = output_dir / "nbs_heterogeneity_relations.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"runs": runs, "correlations": correlations},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.run_dirs):
        raise ValueError("--labels must contain exactly one label per run directory")
    labels = args.labels or [path.parent.name for path in args.run_dirs]
    runs = [load_run(path, label) for path, label in zip(args.run_dirs, labels)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "nbs_heterogeneity_relations.png"
    correlations = plot(runs, figure_path)
    csv_path, json_path = write_outputs(runs, correlations, args.output_dir)
    print(f"Figure saved at {figure_path}")
    print(f"CSV saved at {csv_path}")
    print(f"JSON saved at {json_path}")


if __name__ == "__main__":
    main()
