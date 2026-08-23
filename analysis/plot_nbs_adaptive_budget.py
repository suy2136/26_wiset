"""Plot relative-threshold NBS gains and the resulting effective rank budget."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_FIELDS = {
    "optimizer_step",
    "event",
    "budget_mode",
    "relative_lambda",
    "reference_gain",
    "stopping_threshold",
    "last_allocated_gain",
    "next_rejected_gain",
    "effective_rank_budget",
    "rank_budget_cap",
    "adaptive_min_budget",
    "stopping_reason",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot adaptive NBS threshold, accepted/rejected gains, and budget."
    )
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="Adaptive NBS")
    return parser.parse_args()


def optional_float(value: str | None) -> float:
    if value is None or value.strip() == "":
        return math.nan
    return float(value)


def read_rounds(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [
            row for row in reader
            if row["event"] == "allocation" and row["budget_mode"] == "adaptive"
        ]
    if not rows:
        raise ValueError(f"No adaptive allocation events found in {path}")

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["optimizer_step"])].append(row)

    rounds = []
    for round_index, step in enumerate(sorted(grouped), start=1):
        first = grouped[step][0]
        rounds.append({
            "round": round_index,
            "optimizer_step": step,
            "relative_lambda": optional_float(first["relative_lambda"]),
            "reference_gain": optional_float(first["reference_gain"]),
            "stopping_threshold": optional_float(first["stopping_threshold"]),
            "last_allocated_gain": optional_float(first["last_allocated_gain"]),
            "next_rejected_gain": optional_float(first["next_rejected_gain"]),
            "effective_rank_budget": int(first["effective_rank_budget"]),
            "rank_budget_cap": int(first["rank_budget_cap"]),
            "adaptive_min_budget": int(first["adaptive_min_budget"]),
            "stopping_reason": first["stopping_reason"],
        })
    return rounds


def positive_xy(rounds: list[dict[str, object]], key: str):
    x_values, y_values = [], []
    for row in rounds:
        value = float(row[key])
        if math.isfinite(value) and value > 0:
            x_values.append(int(row["round"]))
            y_values.append(value)
    return x_values, y_values


def main() -> None:
    args = parse_args()
    rounds = read_rounds(args.diagnostics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_values = [int(row["round"]) for row in rounds]
    budgets = [int(row["effective_rank_budget"]) for row in rounds]
    cap = int(rounds[0]["rank_budget_cap"])
    floor = int(rounds[0]["adaptive_min_budget"])
    tau = float(rounds[0]["relative_lambda"])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.suptitle(f"{args.label}: adaptive NBS budget (relative lambda={tau:g})")

    budget_axis = axes[0, 0]
    budget_axis.plot(x_values, budgets, marker="o", color="#1f77b4")
    budget_axis.axhline(cap, linestyle="--", color="#d62728", label=f"cap={cap}")
    budget_axis.axhline(floor, linestyle=":", color="#2ca02c", label=f"floor={floor}")
    budget_axis.set(title="Effective rank budget", xlabel="Allocation round", ylabel="Total rank")
    budget_axis.legend()
    budget_axis.grid(alpha=0.25)

    gain_axis = axes[0, 1]
    gain_styles = (
        ("reference_gain", "Reference gain", "#1f77b4", "-"),
        ("stopping_threshold", "Stopping threshold", "#d62728", "--"),
        ("last_allocated_gain", "Last accepted gain", "#2ca02c", "-"),
        ("next_rejected_gain", "Next rejected gain", "#ff7f0e", ":"),
    )
    for key, label, color, linestyle in gain_styles:
        gain_x, gain_y = positive_xy(rounds, key)
        if gain_x:
            gain_axis.plot(
                gain_x, gain_y, marker="o", markersize=3,
                label=label, color=color, linestyle=linestyle,
            )
    gain_axis.set_yscale("log")
    gain_axis.set(
        title="Marginal Nash gain and relative threshold",
        xlabel="Allocation round",
        ylabel="Gain (log scale)",
    )
    gain_axis.legend()
    gain_axis.grid(alpha=0.25, which="both")

    ratio_axis = axes[1, 0]
    ratios = [budget / cap for budget in budgets]
    ratio_axis.plot(x_values, ratios, marker="o", color="#9467bd")
    ratio_axis.set_ylim(0, 1.05)
    ratio_axis.set(
        title="Budget retained relative to cap",
        xlabel="Allocation round",
        ylabel="Effective budget / cap",
    )
    ratio_axis.grid(alpha=0.25)

    reason_axis = axes[1, 1]
    reasons = Counter(str(row["stopping_reason"]) for row in rounds)
    labels = list(reasons)
    counts = [reasons[label] for label in labels]
    reason_axis.bar(range(len(labels)), counts, color="#8c564b")
    reason_axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    reason_axis.set(
        title="Adaptive stopping reasons",
        ylabel="Allocation rounds",
    )
    reason_axis.grid(alpha=0.2, axis="y")

    figure.tight_layout()
    figure_path = args.output_dir / "nbs_adaptive_budget_trajectory.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    csv_path = args.output_dir / "nbs_adaptive_budget_rounds.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rounds[0]))
        writer.writeheader()
        writer.writerows(rounds)

    summary = {
        "diagnostics": str(args.diagnostics),
        "label": args.label,
        "relative_lambda": tau,
        "allocation_rounds": len(rounds),
        "rank_budget_cap": cap,
        "adaptive_min_budget": floor,
        "effective_budget_min": min(budgets),
        "effective_budget_max": max(budgets),
        "effective_budget_final": budgets[-1],
        "effective_budget_mean": sum(budgets) / len(budgets),
        "stopping_reason_counts": dict(reasons),
    }
    summary_path = args.output_dir / "nbs_adaptive_budget_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Saved: {figure_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
