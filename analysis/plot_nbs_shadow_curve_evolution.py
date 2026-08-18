"""Compare pre-mask and post-training NBS shadow-spectrum utility curves.

New runs save exact ``pre_mask_initialization`` and ``post_training``
snapshots.  Older runs remain supported by falling back to their earliest and
latest validation snapshots, which are explicitly labelled as approximations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot earliest-vs-latest allocation-conditioned shadow utility curves."
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


def quantile(values: Iterable[float], value: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=float), value))


def rank_statistics(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        gain = float(row["marginal_utility_gain"])
        if math.isfinite(gain):
            grouped[int(float(row["current_rank"]))].append(gain)
    result: dict[int, dict[str, float]] = {}
    for rank, values in sorted(grouped.items()):
        result[rank] = {
            "count": float(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q25": quantile(values, 0.25),
            "q75": quantile(values, 0.75),
        }
    return result


def load_run(run_dir: Path, label: str) -> dict[str, Any]:
    curve_path = run_dir / "diminishing_gain" / "nbs_marginal_gain_by_rank.csv"
    report_path = run_dir / "checkpoint_nbs_report" / "report_summary.json"
    if not curve_path.exists():
        raise FileNotFoundError(curve_path)
    rows = read_csv(curve_path)
    by_snapshot: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_snapshot[row["snapshot"]].append(row)
    if not by_snapshot:
        raise ValueError(f"No snapshots in {curve_path}")

    def snapshot_kind(snapshot_rows: list[dict[str, str]]) -> str:
        return snapshot_rows[0].get("snapshot_kind", "validation") or "validation"

    ordered = sorted(
        by_snapshot.items(),
        key=lambda item: (
            int(float(item[1][0]["validation_event"])),
            int(float(item[1][0]["optimizer_step"])),
            item[0],
        ),
    )
    pre_mask = next(
        (item for item in ordered if snapshot_kind(item[1]) == "pre_mask_initialization"),
        None,
    )
    post_training = next(
        (item for item in reversed(ordered) if snapshot_kind(item[1]) == "post_training"),
        None,
    )
    earliest_item = pre_mask or ordered[0]
    latest_item = post_training or ordered[-1]
    selected = {}
    for name, (snapshot_name, event_rows) in (
        ("earliest", earliest_item), ("latest", latest_item)
    ):
        event = int(float(event_rows[0]["validation_event"]))
        selected[name] = {
            "snapshot": snapshot_name,
            "snapshot_kind": snapshot_kind(event_rows),
            "validation_event": event,
            "optimizer_step": max(int(float(row["optimizer_step"])) for row in event_rows),
            "stats": rank_statistics(event_rows),
        }
    report = read_json(report_path)
    return {
        "label": label,
        "run_dir": run_dir,
        "budget": int(report["final_total_rank"]),
        "mean_rank": float(report["final_rank_mean"]),
        **selected,
    }


def positive_curve(stats: dict[int, dict[str, float]]) -> tuple[np.ndarray, ...]:
    ranks = np.asarray(list(stats), dtype=float)
    median = np.asarray([stats[rank]["median"] for rank in stats], dtype=float)
    q25 = np.asarray([stats[rank]["q25"] for rank in stats], dtype=float)
    q75 = np.asarray([stats[rank]["q75"] for rank in stats], dtype=float)
    valid = median > 0
    floor = np.finfo(float).tiny
    return ranks[valid], median[valid], np.maximum(q25[valid], floor), q75[valid]


def plot_run(axis: Any, run: dict[str, Any]) -> None:
    styles = {
        "earliest": {"color": "#4c78a8", "linestyle": "--"},
        "latest": {"color": "#e45756", "linestyle": "-"},
    }
    for name in ("earliest", "latest"):
        curve = run[name]
        ranks, median, q25, q75 = positive_curve(curve["stats"])
        if curve["snapshot_kind"] == "pre_mask_initialization":
            label = "Pre-mask initialization"
        elif curve["snapshot_kind"] == "post_training":
            label = f"Post-training (opt step {curve['optimizer_step']})"
        else:
            label = (
                f"{name.capitalize()} saved: V{curve['validation_event']} "
                f"(opt step {curve['optimizer_step']})"
            )
        axis.plot(
            ranks, median, label=label, linewidth=2.1,
            color=styles[name]["color"], linestyle=styles[name]["linestyle"],
        )
        axis.fill_between(ranks, q25, q75, color=styles[name]["color"], alpha=0.13)
    axis.axvline(
        run["mean_rank"], color="#555555", linestyle=":", linewidth=1.4,
        label=f"Final mean rank = {run['mean_rank']:g}",
    )
    axis.set_yscale("log")
    axis.set(
        title=f"{run['label']}  (budget={run['budget']})",
        xlabel="Current rank r (gain for r → r+1)",
        ylabel="Median log(U(r+1) / U(r))",
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)


def write_summary_csv(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "nbs_pre_mask_vs_post_training_by_rank.csv"
    fields = [
        "label", "curve", "snapshot", "snapshot_kind", "validation_event",
        "optimizer_step", "rank",
        "count", "mean", "median", "q25", "q75",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for curve_name in ("earliest", "latest"):
                curve = run[curve_name]
                for rank, stats in curve["stats"].items():
                    writer.writerow({
                        "label": run["label"],
                        "curve": curve_name,
                        "snapshot": curve["snapshot"],
                        "snapshot_kind": curve["snapshot_kind"],
                        "validation_event": curve["validation_event"],
                        "optimizer_step": curve["optimizer_step"],
                        "rank": rank,
                        **stats,
                    })
    return path


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.run_dirs):
        raise ValueError("--labels must contain exactly one label per run directory")
    labels = args.labels or [path.parent.name for path in args.run_dirs]
    runs = [load_run(path, label) for path, label in zip(args.run_dirs, labels)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 3, figsize=(19, 11), sharex=True, sharey=True, constrained_layout=True)
    figure.suptitle(
        "NBS spectrum: pre-mask initialization vs post-training shadow",
        fontsize=16,
    )
    for axis, run in zip(axes.flat, runs):
        plot_run(axis, run)
    if len(runs) < len(axes.flat):
        note_axis = axes.flat[-1]
        note_axis.axis("off")
        exact_pairs = all(
            run["earliest"]["snapshot_kind"] == "pre_mask_initialization"
            and run["latest"]["snapshot_kind"] == "post_training"
            for run in runs
        )
        if exact_pairs:
            note = (
                "Exact snapshots\n\n"
                "Blue = all physical components before\n"
                "the first rank mask.\n\n"
                "Red = spectral_shadow after training."
            )
        else:
            note = (
                "Compatibility fallback\n\n"
                "Runs without the new exact snapshots use\n"
                "their earliest/latest validation snapshots.\n\n"
                "Those fallback curves are allocation-conditioned."
            )
        note_axis.text(0.04, 0.78, note, va="top", ha="left", fontsize=13)

    figure_path = args.output_dir / "nbs_pre_mask_vs_post_training.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    csv_path = write_summary_csv(runs, args.output_dir)
    print(f"Figure saved at {figure_path}")
    print(f"Summary data saved at {csv_path}")


if __name__ == "__main__":
    main()
