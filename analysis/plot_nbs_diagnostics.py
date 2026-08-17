"""Visualize NBS allocator statistics and rank trajectories from CSV logs.

Examples
--------
Single run::

    python analysis/plot_nbs_diagnostics.py \
        --diagnostics path/to/nbs_rank_diagnostics.csv \
        --labels nbs_v2 --output-dir path/to/figures

Multiple runs additionally produce a comparison figure::

    python analysis/plot_nbs_diagnostics.py \
        --diagnostics path/to/v2.csv path/to/v3.csv \
        --labels nbs_v2 nbs_v3 --output-dir comparison_figures
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_FIELDS = {
    "optimizer_step", "phase", "event", "layer_name", "rank",
    "sensitivity", "alpha", "utility", "next_marginal_gain",
    "min_rank", "max_rank", "at_min_rank", "at_max_rank",
    "rank_delta", "total_rank", "rank_budget",
}
MODULE_COLORS = {"q_proj": "#1f77b4", "v_proj": "#ff7f0e"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot NBS rank-allocation diagnostics and trajectories."
    )
    parser.add_argument(
        "--diagnostics", nargs="+", type=Path, required=True,
        help="One or more nbs_rank_diagnostics.csv files.",
    )
    parser.add_argument(
        "--labels", nargs="*",
        help="Optional labels matching --diagnostics order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def optional_float(value: str | None) -> float:
    if value is None or value.strip() == "":
        return math.nan
    return float(value)


def inferred_label(path: Path) -> str:
    # Standard layout: .../<variant>/<run_id>/nbs_rank_diagnostics.csv
    if path.parent.parent.name:
        return path.parent.parent.name
    return path.stem


def safe_name(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return value.strip("_") or "nbs"


def module_coordinates(row: dict[str, str]) -> tuple[int, int, str]:
    layer_text = row.get("transformer_layer_index", "").strip()
    if layer_text:
        layer_index = int(float(layer_text))
    else:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", row["layer_name"])
        layer_index = int(match.group(1)) if match else 10**9
    module_type = row.get("module_type", "").strip() or row["layer_name"].rsplit(".", 1)[-1]
    module_order = {"q_proj": 0, "v_proj": 1}.get(module_type, 99)
    return layer_index, module_order, module_type


def read_run(path: Path, label: str) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing diagnostic columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No diagnostic rows found in {path}")

    events: OrderedDict[tuple[int, str], list[dict[str, str]]] = OrderedDict()
    module_meta: dict[str, tuple[int, int, str]] = {}
    for row in rows:
        key = (int(row["optimizer_step"]), row["event"])
        events.setdefault(key, []).append(row)
        module_meta[row["layer_name"]] = module_coordinates(row)
    modules = sorted(module_meta, key=lambda name: (*module_meta[name], name))
    event_keys = list(events)

    summaries = []
    for step, event in event_keys:
        group = events[(step, event)]
        summary: dict[str, Any] = {
            "optimizer_step": step,
            "event": event,
            "phase": group[0]["phase"],
            "total_rank": int(group[0]["total_rank"]),
            "rank_budget": int(group[0]["rank_budget"]),
            # Moving one unit from A to B creates two absolute rank deltas.
            "reallocated_rank_units": 0.5 * sum(abs(int(row["rank_delta"])) for row in group),
            "changed_modules": sum(int(row["rank_delta"]) != 0 for row in group),
            "at_min_count": sum(int(row["at_min_rank"]) for row in group),
            "at_max_count": sum(int(row["at_max_rank"]) for row in group),
        }
        for module_type in ("q_proj", "v_proj"):
            selected = [row for row in group if module_coordinates(row)[2] == module_type]
            if not selected:
                continue
            summary[f"{module_type}_rank_total"] = sum(int(row["rank"]) for row in selected)
            for field in (
                "sensitivity", "alpha", "utility", "next_utility_increment",
                "next_marginal_utility_gain", "next_marginal_gain",
            ):
                values = [optional_float(row.get(field)) for row in selected]
                finite = [value for value in values if math.isfinite(value)]
                summary[f"{module_type}_{field}_mean"] = (
                    sum(finite) / len(finite) if finite else math.nan
                )
        summaries.append(summary)

    rank_lookup = {
        ((int(row["optimizer_step"]), row["event"]), row["layer_name"]): int(row["rank"])
        for row in rows
    }
    rank_matrix = np.array([
        [rank_lookup.get((event_key, module), np.nan) for event_key in event_keys]
        for module in modules
    ], dtype=float)
    return {
        "path": path,
        "label": label,
        "rows": rows,
        "events": events,
        "event_keys": event_keys,
        "modules": modules,
        "module_meta": module_meta,
        "summaries": summaries,
        "rank_matrix": rank_matrix,
    }


def sparse_event_ticks(event_keys: list[tuple[int, str]], maximum: int = 12) -> list[int]:
    if len(event_keys) <= maximum:
        return list(range(len(event_keys)))
    chosen = set(np.linspace(0, len(event_keys) - 1, maximum, dtype=int).tolist())
    for index, (_, event) in enumerate(event_keys):
        if event != "allocation":
            chosen.add(index)
    return sorted(chosen)


def series(summaries: list[dict[str, Any]], key: str) -> list[float]:
    return [float(summary.get(key, math.nan)) for summary in summaries]


def plot_two_module_lines(axis: Any, summaries: list[dict[str, Any]], field: str,
                          title: str, ylabel: str) -> None:
    steps = series(summaries, "optimizer_step")
    for module_type in ("q_proj", "v_proj"):
        axis.plot(
            steps,
            series(summaries, f"{module_type}_{field}_mean"),
            label=module_type,
            color=MODULE_COLORS[module_type],
            linewidth=1.7,
        )
    axis.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.legend()


def plot_run_overview(run: dict[str, Any], output_dir: Path) -> Path:
    summaries = run["summaries"]
    event_keys = run["event_keys"]
    modules = run["modules"]
    module_meta = run["module_meta"]
    rank_matrix = run["rank_matrix"]

    figure = plt.figure(figsize=(18, 23), constrained_layout=True)
    grid = figure.add_gridspec(5, 2, height_ratios=(1.8, 1, 1, 1, 1))
    figure.suptitle(f"{run['label']} — NBS allocator diagnostics", fontsize=16)

    heatmap_axis = figure.add_subplot(grid[0, :])
    image = heatmap_axis.imshow(rank_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    tick_indices = sparse_event_ticks(event_keys)
    heatmap_axis.set_xticks(tick_indices)
    heatmap_axis.set_xticklabels(
        [f"{event_keys[index][0]}\n{event_keys[index][1]}" for index in tick_indices],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    heatmap_axis.set_yticks(range(len(modules)))
    heatmap_axis.set_yticklabels([
        (
            f"L{module_meta[name][0]} {module_meta[name][2]}"
            if module_meta[name][0] != 10**9 else name
        )
        for name in modules
    ], fontsize=7)
    heatmap_axis.set(title="Rank trajectory", xlabel="Optimizer step / event", ylabel="LoRA module")
    figure.colorbar(image, ax=heatmap_axis, label="Allocated rank", shrink=0.85)

    rank_axis = figure.add_subplot(grid[1, 0])
    steps = series(summaries, "optimizer_step")
    for module_type in ("q_proj", "v_proj"):
        rank_axis.plot(
            steps,
            series(summaries, f"{module_type}_rank_total"),
            label=module_type,
            color=MODULE_COLORS[module_type],
            linewidth=1.8,
        )
    rank_axis.set(title="Rank budget by projection", xlabel="Optimizer step", ylabel="Total allocated rank")
    rank_axis.grid(alpha=0.25)
    rank_axis.legend()

    churn_axis = figure.add_subplot(grid[1, 1])
    churn_axis.bar(steps, series(summaries, "reallocated_rank_units"), width=max(1, max(steps) * 0.006))
    churn_axis.set(title="Rank reallocation churn", xlabel="Optimizer step", ylabel="Rank units moved")
    churn_axis.grid(axis="y", alpha=0.25)

    sensitivity_axis = figure.add_subplot(grid[2, 0])
    plot_two_module_lines(
        sensitivity_axis, summaries, "sensitivity", "Bias-corrected sensitivity", "Mean sensitivity"
    )
    positive_sensitivity = [value for value in sensitivity_axis.lines[0].get_ydata() if value > 0]
    if positive_sensitivity:
        sensitivity_axis.set_yscale("log")

    alpha_axis = figure.add_subplot(grid[2, 1])
    plot_two_module_lines(alpha_axis, summaries, "alpha", "Bargaining weight", "Mean alpha")

    utility_axis = figure.add_subplot(grid[3, 0])
    plot_two_module_lines(utility_axis, summaries, "utility", "Normalized spectral utility", "Mean utility")

    gain_axis = figure.add_subplot(grid[3, 1])
    plot_two_module_lines(
        gain_axis, summaries, "next_marginal_gain", "Next marginal Nash gain", "Mean gain"
    )
    if any(
        math.isfinite(value)
        for module_type in ("q_proj", "v_proj")
        for value in series(
            summaries, f"{module_type}_next_marginal_utility_gain_mean"
        )
    ):
        for module_type in ("q_proj", "v_proj"):
            gain_axis.plot(
                steps,
                series(
                    summaries,
                    f"{module_type}_next_marginal_utility_gain_mean",
                ),
                label=f"{module_type} utility-only",
                color=MODULE_COLORS[module_type],
                linestyle="--",
                linewidth=1.2,
            )
        gain_axis.legend()
    gain_axis.set_yscale("symlog", linthresh=1e-10)

    boundary_axis = figure.add_subplot(grid[4, 0])
    boundary_axis.plot(steps, series(summaries, "at_min_count"), label="At minimum rank")
    boundary_axis.plot(steps, series(summaries, "at_max_count"), label="At maximum rank")
    boundary_axis.set(title="Bound saturation", xlabel="Optimizer step", ylabel="Number of modules")
    boundary_axis.grid(alpha=0.25)
    boundary_axis.legend()

    histogram_axis = figure.add_subplot(grid[4, 1])
    final_ranks = run["rank_matrix"][:, -1]
    bins = np.arange(np.nanmin(final_ranks) - 0.5, np.nanmax(final_ranks) + 1.5)
    histogram_axis.hist(final_ranks[np.isfinite(final_ranks)], bins=bins, color="#4c78a8", alpha=0.9)
    histogram_axis.set(title="Training-end rank distribution", xlabel="Allocated rank", ylabel="Modules")
    histogram_axis.grid(axis="y", alpha=0.25)

    output_path = output_dir / f"{safe_name(run['label'])}_nbs_diagnostics.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_final_layer_ranks(run: dict[str, Any], output_dir: Path) -> Path | None:
    final_rows = list(run["events"].values())[-1]
    by_layer: dict[int, dict[str, int]] = {}
    for row in final_rows:
        layer_index, _, module_type = module_coordinates(row)
        if layer_index == 10**9 or module_type not in MODULE_COLORS:
            continue
        by_layer.setdefault(layer_index, {})[module_type] = int(row["rank"])
    if not by_layer:
        return None
    layers = sorted(by_layer)
    x = np.arange(len(layers))
    width = 0.4
    figure, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    for offset, module_type in ((-width / 2, "q_proj"), (width / 2, "v_proj")):
        axis.bar(
            x + offset,
            [by_layer[layer].get(module_type, 0) for layer in layers],
            width=width,
            label=module_type,
            color=MODULE_COLORS[module_type],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(layers)
    axis.set(
        title=f"{run['label']} — training-end ranks by Transformer layer",
        xlabel="Transformer layer index",
        ylabel="Allocated rank",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    output_path = output_dir / f"{safe_name(run['label'])}_final_layer_ranks.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def write_event_summary(run: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / f"{safe_name(run['label'])}_nbs_event_summary.csv"
    fieldnames = list(run["summaries"][0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(run["summaries"])
    return path


def plot_comparison(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    figure.suptitle("NBS allocator diagnostics comparison", fontsize=16)
    all_final_ranks = np.concatenate([
        run["rank_matrix"][:, -1][np.isfinite(run["rank_matrix"][:, -1])]
        for run in runs
    ])
    common_bins = np.arange(
        np.nanmin(all_final_ranks) - 0.5,
        np.nanmax(all_final_ranks) + 1.5,
    )

    for run in runs:
        summaries = run["summaries"]
        steps = series(summaries, "optimizer_step")
        churn = series(summaries, "reallocated_rank_units")
        axes[0, 0].plot(steps, np.cumsum(churn), label=run["label"], linewidth=1.8)
        q_total = series(summaries, "q_proj_rank_total")
        total = series(summaries, "total_rank")
        q_share = [100.0 * q / budget for q, budget in zip(q_total, total)]
        axes[0, 1].plot(steps, q_share, label=run["label"], linewidth=1.8)

        final_ranks = run["rank_matrix"][:, -1]
        axes[1, 0].hist(
            final_ranks[np.isfinite(final_ranks)], bins=common_bins, alpha=0.45,
            label=run["label"], histtype="stepfilled",
        )

        final_group = list(run["events"].values())[-1]
        by_layer: dict[int, list[int]] = {}
        for row in final_group:
            layer_index, _, _ = module_coordinates(row)
            if layer_index != 10**9:
                by_layer.setdefault(layer_index, []).append(int(row["rank"]))
        layers = sorted(by_layer)
        axes[1, 1].plot(
            layers,
            [sum(by_layer[layer]) / len(by_layer[layer]) for layer in layers],
            label=run["label"],
            marker="o",
            markersize=3,
            linewidth=1.5,
        )

    axes[0, 0].set(title="Cumulative rank churn", xlabel="Optimizer step", ylabel="Cumulative rank units moved")
    axes[0, 1].set(title="q_proj budget share", xlabel="Optimizer step", ylabel="Share of total rank (%)")
    axes[1, 0].set(title="Training-end rank distribution", xlabel="Allocated rank", ylabel="Modules")
    axes[1, 1].set(title="Training-end mean rank by layer", xlabel="Transformer layer index", ylabel="Mean q/v rank")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend()
    path = output_dir / "nbs_diagnostics_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.diagnostics):
        raise ValueError("--labels must contain exactly one label per diagnostics file")
    labels = args.labels or [inferred_label(path) for path in args.diagnostics]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [read_run(path, label) for path, label in zip(args.diagnostics, labels)]

    for run in runs:
        print(f"Overview saved at {plot_run_overview(run, args.output_dir)}")
        layer_path = plot_final_layer_ranks(run, args.output_dir)
        if layer_path is not None:
            print(f"Layer-rank figure saved at {layer_path}")
        print(f"Event summary saved at {write_event_summary(run, args.output_dir)}")
    if len(runs) > 1:
        print(f"Comparison saved at {plot_comparison(runs, args.output_dir)}")


if __name__ == "__main__":
    main()
