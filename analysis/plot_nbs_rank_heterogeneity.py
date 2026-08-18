"""Compare final NBS rank-allocation heterogeneity across experiments.

For final layer ranks r_l and mean rank R/L, this report computes

* population variance: (1/L) * sum_l (r_l - R/L)^2
* mean absolute deviation: (1/L) * sum_l |r_l - R/L|

The final ``final_nbs_model`` diagnostics rows are preferred.  Older runs that
do not contain that event fall back to the latest allocation event.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final NBS rank-allocation variance and mean absolute deviation."
    )
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_final_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], str, int]:
    final_rows = [row for row in rows if row.get("event") == "final_nbs_model"]
    if final_rows:
        step = max(int(float(row["optimizer_step"])) for row in final_rows)
        selected = [
            row for row in final_rows
            if int(float(row["optimizer_step"])) == step
        ]
        return selected, "final_nbs_model", step

    allocation_rows = [row for row in rows if row.get("event") == "allocation"]
    if not allocation_rows:
        raise ValueError("Diagnostics contain neither final_nbs_model nor allocation rows")
    step = max(int(float(row["optimizer_step"])) for row in allocation_rows)
    selected = [
        row for row in allocation_rows
        if int(float(row["optimizer_step"])) == step
    ]
    return selected, "latest_allocation", step


def load_run(run_dir: Path, label: str) -> dict[str, Any]:
    path = run_dir / "nbs_rank_diagnostics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    final_rows, source_event, optimizer_step = select_final_rows(read_csv(path))

    # One diagnostic row is expected per adapted module.  Keeping the last row
    # makes the calculation robust to an accidentally duplicated final event.
    rank_by_layer = {
        row["layer_name"]: int(float(row["rank"])) for row in final_rows
    }
    if not rank_by_layer:
        raise ValueError(f"No layer ranks found in {path}")
    ranks = np.asarray(list(rank_by_layer.values()), dtype=float)
    mean_rank = float(np.mean(ranks))
    deviations = ranks - mean_rank
    variance = float(np.mean(np.square(deviations)))
    std = float(np.sqrt(variance))
    mad = float(np.mean(np.abs(deviations)))
    return {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "source_event": source_event,
        "optimizer_step": optimizer_step,
        "module_count": int(ranks.size),
        "total_rank": int(np.sum(ranks)),
        "mean_rank": mean_rank,
        "rank_variance": variance,
        "rank_std": std,
        "mean_absolute_deviation": mad,
        "normalized_mad": mad / mean_rank if mean_rank else 0.0,
        "min_rank": int(np.min(ranks)),
        "max_rank": int(np.max(ranks)),
        "unique_rank_count": int(np.unique(ranks).size),
    }


def annotate_bars(axis: Any, bars: Any, digits: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            f"{value:.{digits}f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot(runs: list[dict[str, Any]], output_path: Path) -> None:
    labels = [run["label"] for run in runs]
    x = np.arange(len(runs))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    figure.suptitle("NBS final rank-allocation heterogeneity", fontsize=15)

    variance_bars = axes[0].bar(x, [run["rank_variance"] for run in runs], color="#4c78a8")
    axes[0].set_xticks(x, labels)
    axes[0].set(
        title="Population variance of layer ranks",
        xlabel="Experiment",
        ylabel="Var(r_l) (rank²)",
    )
    axes[0].grid(axis="y", alpha=0.25)
    annotate_bars(axes[0], variance_bars)

    mad_bars = axes[1].bar(
        x, [run["mean_absolute_deviation"] for run in runs], color="#f58518"
    )
    axes[1].set_xticks(x, labels)
    axes[1].set(
        title="Mean distance from global mean rank R/L",
        xlabel="Experiment",
        ylabel="Mean |r_l − R/L| (rank)",
    )
    axes[1].grid(axis="y", alpha=0.25)
    annotate_bars(axes[1], mad_bars)

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_outputs(runs: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "nbs_rank_heterogeneity.csv"
    json_path = output_dir / "nbs_rank_heterogeneity.json"
    fields = list(runs[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(runs, handle, indent=2, ensure_ascii=False)
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.run_dirs):
        raise ValueError("--labels must contain exactly one label per run directory")
    labels = args.labels or [path.parent.name for path in args.run_dirs]
    runs = [load_run(path, label) for path, label in zip(args.run_dirs, labels)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = args.output_dir / "nbs_rank_heterogeneity.png"
    plot(runs, figure_path)
    csv_path, json_path = write_outputs(runs, args.output_dir)
    print(f"Figure saved at {figure_path}")
    print(f"CSV saved at {csv_path}")
    print(f"JSON saved at {json_path}")


if __name__ == "__main__":
    main()
