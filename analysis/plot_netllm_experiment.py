"""Create a durable summary and plot for one NetLLM experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+),\s*global_step\s+(?P<step>\d+),\s*average loss:\s*"
    r"(?P<loss>[-+0-9.eE]+)"
)
VALID_RE = re.compile(r"Valid loss\s+(?P<loss>[-+0-9.eE]+)")
TEACHER_FORCING_VALID_RE = re.compile(
    r"Teacher-forcing validation loss\s+(?P<loss>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("nbs", "nbs_v2", "nbs_v3", "nbs_v4", "nbs_v5",
                 "nbs_v6", "nbs_v7", "nbs_v8", "nbs_v9", "nbs_v10",
                 "nbs_v11", "nbs_v12", "nbs_v12_repeat", "nbs_v13",
                 "nbs_v14", "nbs_v15",
                 "uniform_r12", "uniform_b736", "adalora_peft_r12", "plain"),
        required=True,
    )
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--result-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allocator-state", type=Path)
    parser.add_argument("--allocator-diagnostics", type=Path)
    parser.add_argument("--latency-json", type=Path)
    parser.add_argument("--display-name", type=str,
                        help="Optional plot/summary title override for an inference mode.")
    parser.add_argument(
        "--checkpoint-role",
        choices=("best", "best_ar", "best_post_nbs", "final_nbs"),
        default="best",
    )
    return parser.parse_args()


def parse_train_log(
    path: Path,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    list[dict[str, float]],
]:
    text = path.read_text(encoding="utf-8", errors="replace")
    train_curve = [
        {
            "epoch": int(match.group("epoch")),
            "step": int(match.group("step")),
            "loss": float(match.group("loss")),
        }
        for match in TRAIN_RE.finditer(text)
    ]
    valid_curve = [
        {"index": index + 1, "loss": float(match.group("loss"))}
        for index, match in enumerate(VALID_RE.finditer(text))
    ]
    teacher_forcing_valid_curve = [
        {"index": index + 1, "loss": float(match.group("loss"))}
        for index, match in enumerate(TEACHER_FORCING_VALID_RE.finditer(text))
    ]
    return train_curve, valid_curve, teacher_forcing_valid_curve


def read_results(path: Path) -> tuple[dict[str, float], list[dict[str, float]]]:
    aggregate = None
    per_pair: list[dict[str, float]] = []
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
                per_pair.append(parsed)
    if aggregate is None:
        finite = [row for row in per_pair if math.isfinite(row["mae"]) and math.isfinite(row["rmse"])]
        if not finite:
            raise ValueError(f"No finite evaluation metrics found in {path}")
        aggregate = {
            "mae": sum(row["mae"] for row in finite) / len(finite),
            "rmse": sum(row["rmse"] for row in finite) / len(finite),
        }
    return aggregate, per_pair


def read_allocator_state(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    import torch

    state = torch.load(path, map_location="cpu")
    ranks = state.get("ranks") or state.get("current_ranks") or state.get("rank_pattern") or {}
    values = [int(value) for value in ranks.values()]
    if not values:
        return {"path": str(path), "layer_count": 0, "ranks": {}}
    histogram: dict[str, int] = {}
    for rank in values:
        histogram[str(rank)] = histogram.get(str(rank), 0) + 1
    return {
        "path": str(path),
        "layer_count": len(values),
        "total_rank": sum(values),
        "minimum_rank": min(values),
        "maximum_rank": max(values),
        "mean_rank": sum(values) / len(values),
        "histogram": histogram,
        "ranks": {str(key): int(value) for key, value in ranks.items()},
    }


def read_allocator_diagnostics(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_latency(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_rank_trajectory(rows: list[dict[str, str]], output_dir: Path) -> Path | None:
    if not rows:
        return None
    event_keys = []
    seen_events = set()
    layer_meta: dict[str, tuple[int, int, str]] = {}
    values = {}
    module_order = {"q_proj": 0, "v_proj": 1}
    for row in rows:
        key = (int(row["optimizer_step"]), row["event"])
        if key not in seen_events:
            seen_events.add(key)
            event_keys.append(key)
        name = row["layer_name"]
        layer_text = row.get("transformer_layer_index", "")
        layer_index = int(layer_text) if layer_text else 10**9
        module_type = row.get("module_type", "")
        layer_meta[name] = (
            layer_index, module_order.get(module_type, 99), module_type
        )
        values[(key, name)] = int(row["rank"])
    layer_names = sorted(layer_meta, key=lambda name: (*layer_meta[name], name))
    matrix = [
        [values.get((event, name), math.nan) for event in event_keys]
        for name in layer_names
    ]
    labels = []
    for name in layer_names:
        layer_index, _, module_type = layer_meta[name]
        labels.append(
            name if layer_index == 10**9 else f"L{layer_index} {module_type}"
        )

    width = max(12, min(28, 0.32 * len(event_keys)))
    figure, axis = plt.subplots(figsize=(width, 14), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_title("NBS rank trajectory")
    axis.set_xlabel("Optimizer step / event")
    axis.set_ylabel("LoRA module")
    axis.set_xticks(range(len(event_keys)))
    axis.set_xticklabels(
        [f"{step}\n{event}" for step, event in event_keys],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=7)
    figure.colorbar(image, ax=axis, label="Allocated rank")
    path = output_dir / "nbs_rank_trajectory.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def finite_losses(curve: list[dict[str, float]]) -> list[dict[str, float]]:
    return [point for point in curve if math.isfinite(point["loss"])]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_curve, valid_curve, teacher_forcing_valid_curve = parse_train_log(args.train_log)
    aggregate, per_pair = read_results(args.result_csv)
    allocator = read_allocator_state(args.allocator_state)
    diagnostic_rows = read_allocator_diagnostics(args.allocator_diagnostics)
    latency = read_latency(args.latency_json)
    train_finite = finite_losses(train_curve)
    valid_finite = finite_losses(valid_curve)
    teacher_forcing_valid_finite = finite_losses(teacher_forcing_valid_curve)
    display_names = {
        "nbs": "NBS-NetLLM",
        "nbs_v2": "NBS-NetLLM v2",
        "nbs_v3": "NBS-NetLLM v3",
        "nbs_v4": "NBS-NetLLM v4",
        "nbs_v5": "NBS-NetLLM v5",
        "nbs_v6": "NBS-NetLLM v6 (min2-max32-budget256)",
        "nbs_v7": "NBS-NetLLM v7 (min4-max32-budget512)",
        "nbs_v8": "NBS-NetLLM v8 (min4-max32-budget768)",
        "nbs_v9": "NBS-NetLLM v9 (min4-max32-budget896)",
        "nbs_v10": "NBS-NetLLM v10 (min4-max32-budget640)",
        "nbs_v11": "NBS-NetLLM v11 (min2-max32-budget768)",
        "nbs_v12": "NBS-NetLLM v12 (min4-max32-budget736)",
        "nbs_v12_repeat": "NBS-NetLLM v12 repeat (min4-max32-budget736, seed1)",
        "nbs_v13": "NBS-NetLLM v13 (min4-max32-budget720, seed1)",
        "nbs_v14": "NBS-NetLLM v14 (budget736, lr1.5e-4, ema0.9)",
        "nbs_v15": "NBS-NetLLM v15 (budget736, lr2e-4, ema0.95)",
        "uniform_r12": "Uniform-rank NetLLM (rank12, budget768)",
        "uniform_b736": "Fixed near-uniform NetLLM (ranks11/12, budget736, seed1)",
        "adalora_peft_r12": "Stock PEFT AdaLoRA r12 + Selector + Speculative",
        "plain": "NetLLM",
    }
    display_name = args.display_name or display_names[args.variant]
    display_title = (
        display_name if args.checkpoint_role == "best"
        else f"{display_name} [{args.checkpoint_role}]"
    )

    summary = {
        "variant": args.variant,
        "display_name": display_name,
        "checkpoint_role": args.checkpoint_role,
        "aggregate_mae": aggregate["mae"],
        "aggregate_rmse": aggregate["rmse"],
        "evaluated_pair_count": len(per_pair),
        "final_reported_train_loss": train_finite[-1]["loss"] if train_finite else None,
        "best_reported_valid_loss": min((point["loss"] for point in valid_finite), default=None),
        "best_reported_teacher_forcing_valid_loss": min(
            (point["loss"] for point in teacher_forcing_valid_finite), default=None
        ),
        "train_curve": train_curve,
        "valid_curve": valid_curve,
        "teacher_forcing_valid_curve": teacher_forcing_valid_curve,
        "allocator": allocator,
        "allocator_diagnostics": {
            "path": str(args.allocator_diagnostics) if args.allocator_diagnostics else None,
            "row_count": len(diagnostic_rows),
            "event_count": len({
                (row.get("optimizer_step"), row.get("event"))
                for row in diagnostic_rows
            }),
        },
        "inference_latency": latency,
        "source_files": {
            "train_log": str(args.train_log),
            "result_csv": str(args.result_csv),
            "allocator_diagnostics": (
                str(args.allocator_diagnostics) if args.allocator_diagnostics else None
            ),
            "latency_json": str(args.latency_json) if args.latency_json else None,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(f"{display_title} training and evaluation")
    if train_finite:
        axes[0, 0].plot(
            [point["step"] for point in train_finite],
            [point["loss"] for point in train_finite],
            marker="o",
        )
    else:
        axes[0, 0].text(0.5, 0.5, "No reported training-loss points", ha="center", va="center")
    axes[0, 0].set(title="Reported training loss", xlabel="Global step", ylabel="Loss")
    axes[0, 0].grid(alpha=0.25)

    if valid_finite:
        axes[0, 1].plot(
            [point["index"] for point in valid_finite],
            [point["loss"] for point in valid_finite],
            marker="o",
            color="#d95f02",
            label="Autoregressive",
        )
    if teacher_forcing_valid_finite:
        axes[0, 1].plot(
            [point["index"] for point in teacher_forcing_valid_finite],
            [point["loss"] for point in teacher_forcing_valid_finite],
            marker="o",
            color="#1b9e77",
            label="Teacher forcing",
        )
    if valid_finite or teacher_forcing_valid_finite:
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, "No validation-loss points", ha="center", va="center")
    axes[0, 1].set(title="Validation loss", xlabel="Validation event", ylabel="Loss")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].bar(
        ["MAE", "RMSE"],
        [aggregate["mae"], aggregate["rmse"]],
        color=["#1b9e77", "#7570b3"],
    )
    axes[1, 0].set_title("Aggregate metrics (lower is better)")
    axes[1, 0].grid(axis="y", alpha=0.25)
    if latency and latency.get("mean_s") is not None:
        axes[1, 0].text(
            0.98, 0.96,
            "Latency: mean {:.1f} ms\np50 {:.1f} ms / p95 {:.1f} ms".format(
                latency["mean_s"] * 1000.0,
                latency["median_s"] * 1000.0,
                latency["p95_s"] * 1000.0,
            ),
            transform=axes[1, 0].transAxes,
            ha="right", va="top", fontsize=9,
        )

    finite_mae = [row["mae"] for row in per_pair if math.isfinite(row["mae"])]
    if finite_mae:
        bins = min(20, max(5, int(math.sqrt(len(finite_mae)))))
        axes[0, 2].hist(finite_mae, bins=bins, color="#66a61e", alpha=0.85)
        axes[0, 2].set(title="Per video-user MAE", xlabel="MAE", ylabel="Count")
    else:
        axes[0, 2].text(0.5, 0.5, "No per-pair metrics", ha="center", va="center")
        axes[0, 2].set_title("Per-pair metrics")

    finite_rmse = [row["rmse"] for row in per_pair if math.isfinite(row["rmse"])]
    if finite_rmse:
        bins = min(20, max(5, int(math.sqrt(len(finite_rmse)))))
        axes[1, 1].hist(finite_rmse, bins=bins, color="#e6ab02", alpha=0.85)
        axes[1, 1].set(title="Per video-user RMSE", xlabel="RMSE", ylabel="Count")
    else:
        axes[1, 1].text(0.5, 0.5, "No per-pair metrics", ha="center", va="center")
        axes[1, 1].set_title("Per-pair metrics")

    if allocator and allocator.get("histogram"):
        rank_items = sorted((int(rank), count) for rank, count in allocator["histogram"].items())
        axes[1, 2].bar([str(rank) for rank, _ in rank_items], [count for _, count in rank_items])
        axes[1, 2].set(title="NBS allocated ranks", xlabel="Rank", ylabel="Layers")
    else:
        if args.variant == "uniform_b736":
            rank_text = "Fixed ranks: 32 x 11 + 32 x 12\nTotal budget = 736"
        else:
            uniform_rank = 12 if args.variant == "uniform_r12" else 32
            rank_text = f"Uniform rank = {uniform_rank}"
        axes[1, 2].text(0.5, 0.5, rank_text, ha="center", va="center")
        axes[1, 2].set_title("LoRA rank allocation")

    figure_path = args.output_dir / f"{args.variant}_training_evaluation.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Summary saved at {summary_path}")
    print(f"Figure saved at {figure_path}")
    trajectory_path = plot_rank_trajectory(diagnostic_rows, args.output_dir)
    if trajectory_path is not None:
        print(f"Rank trajectory saved at {trajectory_path}")


if __name__ == "__main__":
    main()
