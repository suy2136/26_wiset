"""Plot module-level evidence linking NBS factors to final allocated ranks.

For every experiment, the 64 LoRA modules are joined by layer name across the
first allocation, last allocation, and final-model diagnostics.  Three panels
are generated:

1. last-allocation bargaining weight alpha_l vs final rank,
2. last-allocation next marginal spectral-utility gain vs final rank,
3. first saved allocation next marginal Nash gain vs final rank.

Panel 3 is deliberately labelled a proxy: legacy diagnostics record the next
gain at the rank selected by the first allocation, not Delta_l(r_l^min) before
that allocation.
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


MODULE_STYLE = {
    "q_proj": {"marker": "o", "color": "#4c78a8"},
    "v_proj": {"marker": "^", "color": "#f58518"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot module-level NBS factors against final allocated rank."
    )
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}={value!r} in {path}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field}={value!r} in {path}")
    return result


def group_allocation_events(rows: list[dict[str, str]]) -> list[tuple[int, list[dict[str, str]]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("event") == "allocation":
            grouped[int(float(row["optimizer_step"]))].append(row)
    if not grouped:
        raise ValueError("Diagnostics contain no allocation events")
    return sorted(grouped.items())


def final_rows(rows: list[dict[str, str]]) -> tuple[int, list[dict[str, str]]]:
    candidates = [row for row in rows if row.get("event") == "final_nbs_model"]
    if not candidates:
        return group_allocation_events(rows)[-1]
    step = max(int(float(row["optimizer_step"])) for row in candidates)
    return step, [
        row for row in candidates if int(float(row["optimizer_step"])) == step
    ]


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        result[order[start:end]] = average_rank
        start = end
    return result


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    return {
        "pearson": pearson(x, y),
        "spearman": pearson(average_tie_ranks(x), average_tie_ranks(y)),
    }


def module_type(row: dict[str, str]) -> str:
    value = row.get("module_type", "").strip()
    if value:
        return value
    return "q_proj" if ".q_proj" in row["layer_name"] else "v_proj"


def load_run(run_dir: Path, label: str) -> dict[str, Any]:
    path = run_dir / "nbs_rank_diagnostics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    allocation_events = group_allocation_events(rows)
    first_step, first_group = allocation_events[0]
    last_step, last_group = allocation_events[-1]
    final_step, final_group = final_rows(rows)

    first = {row["layer_name"]: row for row in first_group}
    last = {row["layer_name"]: row for row in last_group}
    final = {row["layer_name"]: row for row in final_group}
    names = sorted(set(first) & set(last) & set(final))
    expected = max(len(first), len(last), len(final))
    if len(names) != expected:
        raise ValueError(
            f"Layer mismatch in {path}: first={len(first)}, last={len(last)}, "
            f"final={len(final)}, joined={len(names)}"
        )

    records = []
    for name in names:
        records.append({
            "label": label,
            "layer_name": name,
            "transformer_layer_index": int(float(final[name]["transformer_layer_index"])),
            "module_type": module_type(final[name]),
            "min_rank": int(float(final[name]["min_rank"])),
            "final_rank": int(float(final[name]["rank"])),
            "first_allocation_step": first_step,
            "last_allocation_step": last_step,
            "final_step": final_step,
            "last_alpha": safe_float(last[name]["alpha"], "alpha", path),
            "last_marginal_utility_gain": safe_float(
                last[name]["next_marginal_utility_gain"],
                "next_marginal_utility_gain", path,
            ),
            "first_next_marginal_gain_proxy": safe_float(
                first[name]["next_marginal_gain"], "next_marginal_gain", path
            ),
            "first_selected_rank": int(float(first[name]["rank"])),
        })
    return {"label": label, "run_dir": str(run_dir.resolve()), "records": records}


def format_corr(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def set_symlog_if_needed(axis: Any, values: np.ndarray) -> None:
    positive = values[values > 0]
    if positive.size and float(np.max(positive)) / max(float(np.min(positive)), 1e-300) > 100:
        axis.set_xscale("symlog", linthresh=max(float(np.min(positive)) * 0.5, 1e-14))


def plot_panel(
    axis: Any,
    records: list[dict[str, Any]],
    field: str,
    xlabel: str,
    title: str,
) -> dict[str, float | None]:
    x = np.asarray([record[field] for record in records], dtype=float)
    y = np.asarray([record["final_rank"] for record in records], dtype=float)
    result = correlations(x, y)
    for kind, style in MODULE_STYLE.items():
        selected = [index for index, record in enumerate(records) if record["module_type"] == kind]
        # Tiny fixed display offset separates q/v points while retaining the
        # integer final rank as the value used in all statistics.
        offset = -0.07 if kind == "q_proj" else 0.07
        axis.scatter(
            x[selected], y[selected] + offset,
            marker=style["marker"], color=style["color"], alpha=0.72,
            s=38, label=kind,
        )
    set_symlog_if_needed(axis, x)
    axis.set(
        title=(
            f"{title}\nPearson={format_corr(result['pearson'])}, "
            f"Spearman={format_corr(result['spearman'])}"
        ),
        xlabel=xlabel,
        ylabel="Final allocated rank",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    return result


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "nbs"


def plot_run(run: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    figure.suptitle(f"{run['label']} — module-level NBS allocation relationships", fontsize=15)
    specs = (
        (
            "last_alpha", "Last-allocation bargaining weight alpha_l",
            "alpha_l vs final rank (last allocation)",
        ),
        (
            "last_marginal_utility_gain", "Last-allocation marginal spectral gain",
            "Marginal spectral gain vs final rank",
        ),
        (
            "first_next_marginal_gain_proxy", "First saved next marginal Nash gain",
            "First-allocation next-gain proxy vs final rank",
        ),
    )
    correlation_results = {}
    for axis, (field, xlabel, title) in zip(axes, specs):
        correlation_results[field] = plot_panel(
            axis, run["records"], field, xlabel, title
        )
    output_path = output_dir / f"{safe_label(run['label'])}_layer_allocation_relations.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return {"figure": str(output_path.resolve()), "correlations": correlation_results}


def write_outputs(runs: list[dict[str, Any]], summaries: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "nbs_layer_allocation_relations.csv"
    records = [record for run in runs for record in run["records"]]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    json_path = output_dir / "nbs_layer_allocation_relation_correlations.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)
    print(f"Module data saved at {csv_path}")
    print(f"Correlation summary saved at {json_path}")


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.run_dirs):
        raise ValueError("--labels must contain exactly one label per run directory")
    labels = args.labels or [path.parent.name for path in args.run_dirs]
    runs = [load_run(path, label) for path, label in zip(args.run_dirs, labels)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run in runs:
        result = plot_run(run, args.output_dir)
        summaries.append({
            "label": run["label"],
            "run_dir": run["run_dir"],
            "module_count": len(run["records"]),
            **result,
            "third_panel_is_proxy": True,
        })
        print(f"Figure saved at {result['figure']}")
    write_outputs(runs, summaries, args.output_dir)


if __name__ == "__main__":
    main()
