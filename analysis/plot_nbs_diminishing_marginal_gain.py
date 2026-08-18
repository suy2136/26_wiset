"""Reconstruct and plot NBS diminishing marginal utility gains from snapshots.

Unlike the diagnostics CSV, each lightweight ``rank_snapshots/*.pt`` file
contains the full ``spectral_shadow`` for every LoRA module.  This lets us hold
one spectral state fixed, rebuild U_l(r) for every feasible rank, and plot the
exact gain for r -> r + 1:

    log(U_l(r + 1) / U_l(r))

Example (PowerShell)::

    python analysis/plot_nbs_diminishing_marginal_gain.py `
      --run-dir experiment_downloads/nbs_v9/20260818_025211 `
      --output-dir experiment_downloads/nbs_v9/20260818_025211/diminishing_gain `
      --label "NBS v9 (min=4, max=32, budget=896)"
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
import torch


MODULE_COLORS = {"q_proj": "#1f77b4", "v_proj": "#ff7f0e"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the exact per-rank NBS marginal utility curves reconstructed "
            "from allocator spectral snapshots."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", default="NBS")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Optional single snapshot. By default all rank_snapshots/*.pt are used.",
    )
    return parser.parse_args()


def module_type(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def transformer_layer(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def load_snapshot(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    required = {"spectral_shadow", "min_ranks", "max_ranks"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"{path} is missing allocator fields: {sorted(missing)}")
    metadata = state.get("snapshot_metadata", {})
    return {
        "path": path,
        "state": state,
        "snapshot_kind": str(metadata.get("snapshot_kind", "validation")),
        "validation_event": int(metadata.get("validation_event", 0)),
        "optimizer_step": int(metadata.get("optimizer_step", state.get("last_step") or 0)),
        "validation_loss": metadata.get("validation_loss"),
    }


def discover_snapshots(run_dir: Path, explicit: Path | None) -> list[dict[str, Any]]:
    paths = [explicit] if explicit else sorted((run_dir / "rank_snapshots").glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No rank snapshots found under {run_dir / 'rank_snapshots'}")
    snapshots = [load_snapshot(path) for path in paths]
    kind_order = {
        "pre_mask_initialization": 0,
        "validation": 1,
        "post_training": 2,
    }
    snapshots.sort(key=lambda item: (
        kind_order.get(item["snapshot_kind"], 1),
        item["validation_event"],
        item["optimizer_step"],
    ))
    return snapshots


def reconstruct_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    state = snapshot["state"]
    eps = float(state.get("eps", 1e-8))
    rows: list[dict[str, Any]] = []

    for name, shadow in state["spectral_shadow"].items():
        min_rank = int(state["min_ranks"][name])
        max_rank = int(state["max_ranks"][name])
        # Reconstruct in float64.  The saved shadow is usually float32, but
        # cumulative utility values become almost identical once the remaining
        # spectral energy is tiny; float32 log-ratios can then show artificial
        # ~1e-7 upward steps that are only rounding noise.
        energy = shadow.detach().double().reshape(-1).abs().square()
        if energy.numel() < max_rank:
            energy = torch.nn.functional.pad(energy, (0, max_rank - energy.numel()))
        energy = torch.sort(energy, descending=True).values[:max_rank]
        total = energy.sum().clamp_min(eps)
        utility = eps + torch.cat((energy.new_zeros(1), torch.cumsum(energy, dim=0))) / total

        gains = []
        for rank in range(min_rank, max_rank):
            gain = float(torch.log(utility[rank + 1] / utility[rank]).item())
            gains.append(gain)
            rows.append({
                "snapshot": snapshot["path"].name,
                "snapshot_kind": snapshot["snapshot_kind"],
                "validation_event": snapshot["validation_event"],
                "optimizer_step": snapshot["optimizer_step"],
                "validation_loss": snapshot["validation_loss"],
                "layer_name": name,
                "transformer_layer_index": transformer_layer(name),
                "module_type": module_type(name),
                "current_rank": rank,
                "next_rank": rank + 1,
                "spectral_component_energy": float(energy[rank].item()),
                "utility_at_rank": float(utility[rank].item()),
                "marginal_utility_gain": gain,
            })

        tolerance = 1e-12
        violations = [
            gains[index + 1] - gains[index]
            for index in range(len(gains) - 1)
            if gains[index + 1] > gains[index] + tolerance
        ]
        snapshot.setdefault("layer_audits", []).append({
            "layer_name": name,
            "comparisons": max(0, len(gains) - 1),
            "violations": len(violations),
            "max_positive_violation": max(violations, default=0.0),
        })
    return rows


def rank_statistics(
    rows: list[dict[str, Any]], field: str = "marginal_utility_gain"
) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if math.isfinite(value):
            grouped[int(row["current_rank"])].append(value)
    result = {}
    for rank, rank_values in sorted(grouped.items()):
        array = np.asarray(rank_values, dtype=float)
        result[rank] = {
            "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
            "mean": float(np.mean(array)),
        }
    return result


def plot_band(axis: Any, stats: dict[int, dict[str, float]], label: str,
              color: str, linewidth: float = 2.2, alpha: float = 0.18) -> None:
    ranks = np.asarray(list(stats), dtype=float)
    medians = np.asarray([stats[rank]["median"] for rank in stats], dtype=float)
    q25 = np.asarray([stats[rank]["q25"] for rank in stats], dtype=float)
    q75 = np.asarray([stats[rank]["q75"] for rank in stats], dtype=float)
    axis.plot(ranks, medians, label=label, color=color, linewidth=linewidth)
    axis.fill_between(ranks, q25, q75, color=color, alpha=alpha)


def configure_gain_axis(axis: Any, title: str) -> None:
    axis.set(
        title=title,
        xlabel="Current rank r (gain is for r → r+1)",
        ylabel="Marginal utility gain  log(U(r+1) / U(r))",
    )
    axis.set_yscale("log")
    axis.grid(alpha=0.25)


def plot_report(
    snapshots: list[dict[str, Any]],
    rows_by_snapshot: list[list[dict[str, Any]]],
    output_dir: Path,
    label: str,
) -> Path:
    latest_rows = rows_by_snapshot[-1]
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    figure.suptitle(f"{label} — diminishing marginal utility gain", fontsize=16)

    all_by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in latest_rows:
        all_by_layer[row["layer_name"]].append(row)
    for layer_rows in all_by_layer.values():
        axes[0, 0].plot(
            [row["current_rank"] for row in layer_rows],
            [row["marginal_utility_gain"] for row in layer_rows],
            color="#9e9e9e",
            alpha=0.18,
            linewidth=0.7,
        )
    plot_band(
        axes[0, 0], rank_statistics(latest_rows), "All modules median ± IQR", "#111111"
    )
    configure_gain_axis(axes[0, 0], "Latest snapshot: all 64 LoRA modules")
    axes[0, 0].legend()

    for kind, color in MODULE_COLORS.items():
        selected = [row for row in latest_rows if row["module_type"] == kind]
        if selected:
            plot_band(axes[0, 1], rank_statistics(selected), f"{kind} median ± IQR", color)
    configure_gain_axis(axes[0, 1], "Latest snapshot by projection")
    axes[0, 1].legend()

    color_map = plt.get_cmap("viridis")
    denominator = max(1, len(snapshots) - 1)
    for index, (snapshot, rows) in enumerate(zip(snapshots, rows_by_snapshot)):
        stats = rank_statistics(rows)
        ranks = list(stats)
        if snapshot["snapshot_kind"] == "pre_mask_initialization":
            label_text = "Pre-mask initialization"
        elif snapshot["snapshot_kind"] == "post_training":
            label_text = f"Post-training / step {snapshot['optimizer_step']}"
        else:
            label_text = f"V{snapshot['validation_event']} / step {snapshot['optimizer_step']}"
        axes[1, 0].plot(
            ranks,
            [stats[rank]["median"] for rank in ranks],
            label=label_text,
            color=color_map(index / denominator),
            linewidth=1.6,
        )
    configure_gain_axis(axes[1, 0], "Median curve across validation snapshots")
    axes[1, 0].legend(fontsize=8, ncol=2)

    events = []
    fractions = []
    for snapshot in snapshots:
        comparisons = sum(item["comparisons"] for item in snapshot["layer_audits"])
        violations = sum(item["violations"] for item in snapshot["layer_audits"])
        events.append(snapshot["validation_event"])
        fractions.append(1.0 if comparisons == 0 else 1.0 - violations / comparisons)
    axes[1, 1].plot(events, fractions, marker="o", linewidth=1.8, color="#2ca02c")
    axes[1, 1].axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    axes[1, 1].set(
        title="Diminishing-gain monotonicity audit",
        xlabel="Validation event",
        ylabel="Fraction of adjacent rank steps non-increasing",
        ylim=(max(0.0, min(fractions) - 0.02), 1.005),
    )
    axes[1, 1].grid(alpha=0.25)

    output_path = output_dir / "nbs_diminishing_marginal_gain.png"
    figure.savefig(output_path, dpi=190)
    plt.close(figure)
    return output_path


def write_curve_csv(rows_by_snapshot: list[list[dict[str, Any]]], output_dir: Path) -> Path:
    rows = [row for snapshot_rows in rows_by_snapshot for row in snapshot_rows]
    output_path = output_dir / "nbs_marginal_gain_by_rank.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_summary(snapshots: list[dict[str, Any]], output_dir: Path) -> Path:
    records = []
    total_comparisons = 0
    total_violations = 0
    for snapshot in snapshots:
        comparisons = sum(item["comparisons"] for item in snapshot["layer_audits"])
        violations = sum(item["violations"] for item in snapshot["layer_audits"])
        total_comparisons += comparisons
        total_violations += violations
        records.append({
            "snapshot": snapshot["path"].name,
            "snapshot_kind": snapshot["snapshot_kind"],
            "validation_event": snapshot["validation_event"],
            "optimizer_step": snapshot["optimizer_step"],
            "adjacent_rank_comparisons": comparisons,
            "monotonicity_violations": violations,
            "non_increasing_fraction": (
                1.0 if comparisons == 0 else 1.0 - violations / comparisons
            ),
            "max_positive_violation": max(
                (item["max_positive_violation"] for item in snapshot["layer_audits"]),
                default=0.0,
            ),
        })
    payload = {
        "definition": "log(U_l(r+1) / U_l(r)) at fixed spectral snapshot",
        "snapshots": records,
        "overall_adjacent_rank_comparisons": total_comparisons,
        "overall_monotonicity_violations": total_violations,
        "overall_non_increasing_fraction": (
            1.0 if total_comparisons == 0
            else 1.0 - total_violations / total_comparisons
        ),
    }
    output_path = output_dir / "nbs_diminishing_gain_summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "diminishing_gain"
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = discover_snapshots(args.run_dir, args.snapshot)
    rows_by_snapshot = [reconstruct_rows(snapshot) for snapshot in snapshots]
    outputs = [
        plot_report(snapshots, rows_by_snapshot, output_dir, args.label),
        write_curve_csv(rows_by_snapshot, output_dir),
        write_summary(snapshots, output_dir),
    ]
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
