"""Read-only three-seed VP evaluation of fixed non-NBS LoRA checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.evaluate_cached_patch_selectors_compact import (
    absolute,
    run_case,
    write_rows,
)
from analysis.evaluate_vp_fast_fallback_fullstack_multiseed import read_safety


SEEDS = (1, 2, 3)
METHODS = (
    ("uniform", "Uniform LoRA", "uniform_checkpoint", 1),
    ("eva", "EVA", "eva_checkpoint", 1),
    ("adalora", "AdaLoRA (preliminary data-seed 2)", "adalora_checkpoint", 2),
    ("shapley", "ShapLoRA (preliminary rank 645)", "shapley_checkpoint", 1),
)
METHOD_BY_NAME = {spec[0]: spec for spec in METHODS}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--methods", nargs="+", choices=tuple(METHOD_BY_NAME),
        default=[spec[0] for spec in METHODS],
        help="Methods to evaluate, in execution order (default: all).",
    )
    for method, _, argument, _ in METHODS:
        result.add_argument(
            f"--{argument.replace('_', '-')}", type=Path,
            help=f"Existing {method} checkpoint directory (read only).",
        )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--latency-warmup-steps", type=int, default=5)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def selected_method_specs(args) -> list[tuple]:
    specs = [METHOD_BY_NAME[name] for name in args.methods]
    for method, _, argument, _ in specs:
        if getattr(args, argument) is None:
            raise ValueError(
                f"--{argument.replace('_', '-')} is required when "
                f"--methods includes {method}"
            )
    return specs


def read_adapter_config(checkpoint: Path) -> dict:
    path = checkpoint / "adapter_config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def active_rank(value) -> int:
    """Return active rank from LoRA integer ranks or AdaLoRA mask lists."""
    if isinstance(value, list):
        if not all(isinstance(item, (bool, int)) for item in value):
            raise TypeError(f"unsupported rank mask values: {value!r}")
        if not all(int(item) in (0, 1) for item in value):
            raise ValueError(f"rank mask must contain only booleans/0/1: {value!r}")
        return sum(bool(item) for item in value)
    if isinstance(value, bool):
        return int(value)
    return int(value)


def checkpoint_description(method: str, checkpoint: Path) -> dict:
    required = [checkpoint / "modules_except_plm.bin"]
    if not any(
        (checkpoint / name).is_file()
        for name in ("adapter_model.bin", "adapter_model.safetensors")
    ):
        raise FileNotFoundError(f"adapter weights absent: {checkpoint}")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    config = read_adapter_config(checkpoint)
    peft_type = str(config.get("peft_type", "")).upper()
    expected = "ADALORA" if method in {"adalora", "shapley"} else "LORA"
    if peft_type != expected:
        raise ValueError(
            f"{method} expects PEFT type {expected}, found {peft_type or 'missing'}"
        )
    rank_pattern = config.get("rank_pattern") or {}
    if rank_pattern:
        active_rank_total = sum(
            active_rank(value) for value in rank_pattern.values()
        )
        rank_pattern_count = len(rank_pattern)
    else:
        target_modules = config.get("target_modules") or []
        # Llama-2 7B has 32 layers; q_proj/v_proj means 64 adapted modules.
        module_count = 32 * len(target_modules)
        active_rank_total = int(config["r"]) * module_count
        rank_pattern_count = 0
    return {
        "method": method,
        "checkpoint": str(checkpoint),
        "peft_type": peft_type,
        "rank": int(config["r"]),
        "init_rank": (
            int(config["init_r"]) if config.get("init_r") is not None else None
        ),
        "rank_pattern_count": rank_pattern_count,
        "active_rank_total": active_rank_total,
        "budget_note": (
            "matched_512" if active_rank_total == 512
            else f"preliminary_nonmatching_{active_rank_total}"
        ),
    }


def write_fixed_rank_config(description: dict, checkpoint: Path,
                            output_dir: Path) -> Path | None:
    config = read_adapter_config(checkpoint)
    rank_pattern = config.get("rank_pattern") or {}
    if description["peft_type"] != "LORA" or not rank_pattern:
        return None
    path = output_dir / f"{description['method']}_rank_pattern.json"
    payload = {
        "rank_pattern": {
            str(key): active_rank(value) for key, value in rank_pattern.items()
        },
        "total_rank_budget": description["active_rank_total"],
        "source_checkpoint": str(checkpoint),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_command(args, description: dict, checkpoint: Path, seed: int,
                  result_dir: Path, rank_config: Path | None) -> list[str]:
    command = [
        sys.executable, "run_plm.py", "--test",
        "--train-dataset", "Jin2022", "--test-dataset", "Jin2022",
        "--evaluation-split", "test",
        "--plm-type", "llama", "--plm-size", "base",
        "--device", args.device, "--device-out", args.device,
        "--fp16", "--vp-fp16-fallback",
        "--rank", str(description["rank"]),
        "--model-path", str(checkpoint),
        "--epochs", "4", "--bs", "1", "--grad-accum-steps", "32",
        "--lr", "0.0002",
        "--seed", str(seed), "--lora-seed", "1", "--data-seed", str(seed),
        "--evaluation-rng-mode", "continuous",
        "--measure-inference-latency",
        "--latency-warmup-steps", str(args.latency_warmup_steps),
        "--save-test-progress-per-steps", "500",
        "--latency-output-path", str(result_dir / "latency.json"),
        "--results-output-dir", str(result_dir),
    ]
    if description["peft_type"] == "ADALORA":
        command.extend([
            "--use-adalora", "--adalora-allocator", "peft",
            "--adalora-init-rank", str(description["init_rank"] or 32),
            "--adalora-allocation-interval", "10",
        ])
    elif rank_config is not None:
        command.extend(["--lora-rank-config", str(rank_config)])
    return command


def summarize(rows: list[dict], method_specs=METHODS) -> list[dict]:
    summaries = []
    for method, label, _, training_data_seed in method_specs:
        group = [
            row for row in rows
            if row.get("method") == method and row.get("status") == "complete"
        ]
        if {int(row["evaluation_seed"]) for row in group} != set(SEEDS):
            raise RuntimeError(f"incomplete three-seed result: {method}")
        item = {
            "method": method,
            "label": label,
            "checkpoint_training_data_seed": training_data_seed,
            "seed_count": len(SEEDS),
            "evaluation_seeds": "1,2,3",
            "evaluation_rng_mode": "continuous",
            "active_rank_total": int(group[0]["active_rank_total"]),
            "budget_note": group[0]["budget_note"],
        }
        for metric in ("mae", "rmse", "latency_mean_ms"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std"] = statistics.stdev(values)
        for metric in (
            "fp16_prescaled_qk_fallback_calls",
            "fp32_attention_fallback_calls",
        ):
            item[f"{metric}_total"] = sum(
                int(row.get(metric, 0)) for row in group
            )
        summaries.append(item)
    return summaries


def plot_summary(rows: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped summary plot", flush=True)
        return
    labels = [row["label"] for row in rows]
    colors = ["#6F63B6", "#E3A52B", "#4E9F6D", "#B45A68"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].barh(
        labels, [row["mae_mean"] for row in rows],
        xerr=[row["mae_std"] for row in rows], color=colors, capsize=4,
    )
    axes[0].invert_yaxis()
    axes[0].set(title="VP MAE, 3-seed mean", xlabel="Degrees; lower is better")
    axes[1].barh(
        labels, [row["latency_mean_ms_mean"] for row in rows],
        xerr=[row["latency_mean_ms_std"] for row in rows],
        color=colors, capsize=4,
    )
    axes[1].invert_yaxis()
    axes[1].set(title="VP latency, 3-seed mean", xlabel="ms; lower is better")
    figure.suptitle("Fixed non-NBS LoRA checkpoints: fast path + safety fallback")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_manifest(args, descriptions: list[dict]) -> None:
    signature = {
        "methods": descriptions,
        "evaluation_seeds": list(SEEDS),
        "evaluation_rng_mode": "continuous",
        "extra_inference_modules": "disabled",
        "numeric_safety": (
            "normal FP16 fast path; prescaled-QK then FP32-attention retries "
            "only after non-finite prediction"
        ),
    }
    path = args.output_dir / "manifest.json"
    if path.is_file() and args.resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError("resume manifest differs from current configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    else:
        path.write_text(
            json.dumps({"signature": signature}, indent=2), encoding="utf-8"
        )


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    method_specs = selected_method_specs(args)
    for _, _, argument, _ in method_specs:
        setattr(args, argument, absolute(getattr(args, argument)))
    args.output_dir = absolute(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    descriptions = []
    checkpoints = {}
    for method, _, argument, _ in method_specs:
        checkpoint = getattr(args, argument)
        checkpoints[method] = checkpoint
        descriptions.append(checkpoint_description(method, checkpoint))
    write_manifest(args, descriptions)

    rows_path = args.output_dir / "per_seed_results.csv"
    rows = []
    if args.resume and rows_path.is_file():
        with rows_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    completed = {
        (row["method"], int(row["evaluation_seed"]))
        for row in rows if row.get("status") == "complete"
    }

    for description in descriptions:
        method = description["method"]
        checkpoint = checkpoints[method]
        rank_config = write_fixed_rank_config(
            description, checkpoint, args.output_dir
        )
        print(
            f"[{method}] PEFT={description['peft_type']} "
            f"active-rank={description['active_rank_total']} "
            f"({description['budget_note']})",
            flush=True,
        )
        for seed in SEEDS:
            if (method, seed) in completed:
                continue
            result_dir = args.output_dir / method / f"seed_{seed}"
            command = build_command(
                args, description, checkpoint, seed, result_dir, rank_config
            )
            if args.dry_run:
                print(json.dumps({"method": method, "seed": seed,
                                  "command": command}))
                continue
            row = {
                **description,
                "evaluation_seed": seed,
                "evaluation_rng_mode": "continuous",
                "status": "failed",
            }
            try:
                row.update(run_case(command, result_dir, args.resume))
                row.update(read_safety(result_dir))
                row["status"] = "complete"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows = [
                old for old in rows
                if not (
                    old.get("method") == method
                    and int(old.get("evaluation_seed", -1)) == seed
                )
            ]
            rows.append(row)
            write_rows(rows_path, rows)
            print(
                f"[{method} seed={seed}] {row['status']} "
                f"MAE={row.get('mae')} latency={row.get('latency_mean_ms')} "
                f"fallbacks={row.get('fp16_prescaled_qk_fallback_calls')}",
                flush=True,
            )
            if row["status"] != "complete":
                raise RuntimeError(row["error"])

    if args.dry_run:
        print(
            f"Dry run: {len(method_specs)} fixed checkpoints x "
            f"{len(SEEDS)} seeds = "
            f"{len(method_specs) * len(SEEDS)} evaluations"
        )
        return
    summaries = summarize(rows, method_specs)
    write_rows(args.output_dir / "three_seed_summary.csv", summaries)
    (args.output_dir / "three_seed_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    plot_summary(summaries, args.output_dir / "three_seed_summary.png")
    print(f"Per-seed results: {rows_path}")
    print(f"Three-seed summary: {args.output_dir / 'three_seed_summary.csv'}")


if __name__ == "__main__":
    main()
