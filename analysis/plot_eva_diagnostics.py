"""Visualize a precomputed EVA rank pattern and retained PCA variance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.eva_initializer import validate_eva_state


MODULE_PATTERN = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\.(?P<kind>q_proj|v_proj)$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def diagnostic_rows(state):
    rows = []
    for name, rank_value in state["rank_pattern"].items():
        match = MODULE_PATTERN.search(name)
        if match is None:
            raise ValueError(f"Cannot parse EVA Llama module name: {name}")
        rank = int(rank_value)
        metrics = state["explained_variance"][name]
        ratio = torch.as_tensor(metrics["ratio"], dtype=torch.float64).flatten()
        retained = float(ratio[:rank].sum().item()) if rank else 0.0
        total = float(ratio.sum().item())
        retained_fraction = retained / total if total > 0 else 0.0
        next_gain = float(ratio[rank].item()) if rank < ratio.numel() else 0.0
        rows.append({
            "layer_name": name,
            "transformer_layer": int(match.group("layer")),
            "module_type": match.group("kind"),
            "rank": rank,
            "retained_explained_variance": retained,
            "retained_explained_variance_fraction": retained_fraction,
            "next_explained_variance_gain": next_gain,
        })
    return sorted(
        rows,
        key=lambda row: (row["transformer_layer"], row["module_type"]),
    )


def summarize(state, rows):
    ranks = [row["rank"] for row in rows]
    retained = [row["retained_explained_variance_fraction"] for row in rows]
    rank_mean = sum(ranks) / len(ranks)
    rank_variance = sum((rank - rank_mean) ** 2 for rank in ranks) / len(ranks)
    return {
        "method": "eva",
        "state_path": str(state.get("source_path", "")),
        "module_count": len(rows),
        "positive_rank_module_count": sum(rank > 0 for rank in ranks),
        "total_rank_budget": int(sum(ranks)),
        "rank_min": min(ranks),
        "rank_max": max(ranks),
        "rank_mean": rank_mean,
        "rank_variance": rank_variance,
        "rank_std": math.sqrt(rank_variance),
        "mean_retained_explained_variance_fraction": sum(retained) / len(retained),
        "min_retained_explained_variance_fraction": min(retained),
        "processed_batches": int(state["processed_batches"]),
        "converged": bool(state["converged"]),
        "metric": state["metric"],
    }


def main():
    args = parse_args()
    state = torch.load(args.state, map_location="cpu")
    validate_eva_state(state)
    if "explained_variance" not in state:
        raise ValueError("EVA state has no explained_variance diagnostics")
    state["source_path"] = str(args.state)
    rows = diagnostic_rows(state)
    summary = summarize(state, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "eva_layer_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "eva_diagnostics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    layers = sorted({row["transformer_layer"] for row in rows})
    by_coordinate = {
        (row["transformer_layer"], row["module_type"]): row for row in rows
    }
    q_ranks = [by_coordinate[(layer, "q_proj")]["rank"] for layer in layers]
    v_ranks = [by_coordinate[(layer, "v_proj")]["rank"] for layer in layers]
    ranks = [row["rank"] for row in rows]
    retained = [row["retained_explained_variance_fraction"] for row in rows]
    next_gains = [row["next_explained_variance_gain"] for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    width = 0.38
    positions = list(range(len(layers)))
    axes[0, 0].bar([x - width / 2 for x in positions], q_ranks, width, label="q_proj")
    axes[0, 0].bar([x + width / 2 for x in positions], v_ranks, width, label="v_proj")
    axes[0, 0].set(
        title="EVA rank allocation across Transformer layers",
        xlabel="Transformer layer",
        ylabel="Allocated rank",
        xticks=positions,
        xticklabels=layers,
    )
    axes[0, 0].legend()

    rank_values = sorted(set(ranks))
    axes[0, 1].bar(
        rank_values,
        [sum(rank == value for rank in ranks) for value in rank_values],
    )
    axes[0, 1].set(title="EVA rank histogram", xlabel="Rank", ylabel="LoRA modules")

    axes[1, 0].hist(retained, bins=min(16, max(5, int(math.sqrt(len(retained))))))
    axes[1, 0].set(
        title="Retained activation explained variance",
        xlabel="Retained fraction",
        ylabel="LoRA modules",
    )

    colors = ["#1f77b4" if row["module_type"] == "q_proj" else "#ff7f0e" for row in rows]
    scatter = axes[1, 1].scatter(ranks, next_gains, c=colors, alpha=0.8)
    del scatter
    if any(value > 0 for value in next_gains):
        axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        title="Rank and next explained-variance gain",
        xlabel="Allocated rank",
        ylabel="Next PCA variance ratio",
    )
    axes[1, 1].text(
        0.02,
        0.98,
        "blue: q_proj\norange: v_proj",
        transform=axes[1, 1].transAxes,
        va="top",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)

    figure.suptitle(
        "EVA diagnostics — budget {}, rank mean {:.2f}, converged {}".format(
            summary["total_rank_budget"], summary["rank_mean"], summary["converged"]
        )
    )
    figure_path = args.output_dir / "eva_rank_diagnostics.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    print(f"EVA diagnostics saved at {figure_path}")
    print(f"EVA layer statistics saved at {csv_path}")
    print(f"EVA summary saved at {summary_path}")


if __name__ == "__main__":
    main()
