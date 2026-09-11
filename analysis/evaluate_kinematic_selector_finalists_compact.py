"""Evaluate VP kinematic-selector finalists with compact NBS-v19.

The validation search directory is read-only.  A compact checkpoint and every
test artifact are written below a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NBS = Path(
    "viewport_prediction/data/ft_plms/llama_base_low_rank_adalora_nbs_v19/"
    "freeze_plm_False/multimodal_none/Jin2022/5Hz/20260821_204908/"
    "his_10_fut_20_ss_15_epochs_4_bs_32_lr_0.0002_seed_1_rank_32_"
    "scheduled_sampling_False/best_ar_model"
)
DEFAULT_PROJECTOR = Path(
    "patch_selection_delivery/model/vp_3condition_best_model"
)
PARAMETER_KEYS = (
    "top_k",
    "velocity_window",
    "horizon_scale",
    "acceleration_weight",
    "uncertainty_deg",
    "uncertainty_growth",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--search-run-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--nbs-checkpoint", type=Path, default=DEFAULT_NBS)
    result.add_argument("--projector-checkpoint", type=Path,
                        default=DEFAULT_PROJECTOR)
    result.add_argument("--device", default="cuda")
    result.add_argument("--latency-warmup-steps", type=int, default=5)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def resolve_from_repo(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_finalists(search_run_dir: Path) -> list[dict[str, object]]:
    path = search_run_dir / "full_validation_results.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        source = list(csv.DictReader(stream))
    rows: list[dict[str, object]] = []
    for row in source:
        if row.get("status") != "complete":
            continue
        rows.append({
            "candidate": int(float(row["candidate"])),
            **{key: float(row[key]) for key in PARAMETER_KEYS},
            "validation_mae": float(row["mae"]),
            "validation_rmse": float(row["rmse"]),
            "validation_latency_mean_ms": float(row["latency_mean_ms"]),
        })
    if len(rows) != 5:
        raise RuntimeError(
            f"expected 5 completed full-validation finalists in {path}, "
            f"found {len(rows)}"
        )
    return rows


def common_command(args: argparse.Namespace, model_path: Path,
                   result_dir: Path) -> list[str]:
    return [
        sys.executable, "run_plm.py", "--test",
        "--train-dataset", "Jin2022", "--test-dataset", "Jin2022",
        "--evaluation-split", "test",
        "--plm-type", "llama", "--plm-size", "base",
        "--device", args.device, "--device-out", args.device,
        "--fp16", "--rank", "32", "--use-adalora",
        "--adalora-allocator", "nbs",
        "--adalora-rank-config",
        "configs/adalora_rank_config_llama7b_min2_max32.json",
        "--adalora-rank-budget", "512", "--adalora-ema-beta", "0.9",
        "--adalora-shadow-update-policy", "legacy",
        "--adalora-allocation-interval", "10",
        "--experiment-tag", "nbs_v19",
        "--model-path", str(model_path), "--evaluation-tag", "best_ar",
        "--nbs-inference-mode", "compact",
        "--epochs", "4", "--bs", "1", "--grad-accum-steps", "32",
        "--lr", "0.0002", "--seed", "1", "--lora-seed", "1",
        "--data-seed", "1", "--measure-inference-latency",
        "--latency-warmup-steps", str(args.latency_warmup_steps),
        "--save-test-progress-per-steps", "500",
        "--latency-output-path", str(result_dir / "latency.json"),
        "--results-output-dir", str(result_dir),
    ]


def build_command(args: argparse.Namespace, model_path: Path,
                  result_dir: Path, config: dict[str, object] | None,
                  compact_output_dir: Path | None = None) -> list[str]:
    command = common_command(args, model_path, result_dir)
    if compact_output_dir is not None:
        command.extend(["--nbs-compact-output-dir", str(compact_output_dir)])
    if config is None:
        return command
    command.extend([
        "--multimodal-mode", "patch-selection",
        "--patch-selector-type", "kinematic",
        "--patch-top-k", f"{config['top_k']:g}",
        "--kinematic-velocity-window", f"{config['velocity_window']:g}",
        "--kinematic-horizon-scale", f"{config['horizon_scale']:g}",
        "--kinematic-acceleration-weight", f"{config['acceleration_weight']:g}",
        "--kinematic-uncertainty-deg", f"{config['uncertainty_deg']:g}",
        "--kinematic-uncertainty-growth", f"{config['uncertainty_growth']:g}",
        "--multimodal-projector-checkpoint", str(args.projector_checkpoint),
        "--inference-tag", "selector",
    ])
    return command


def aggregate_result(directory: Path) -> dict[str, float]:
    result_files = sorted(
        path for path in directory.glob("*_results.csv")
        if "_partial_" not in path.name and "_per_sample_" not in path.name
    )
    if not result_files:
        raise FileNotFoundError(f"aggregate results CSV absent in {directory}")
    with result_files[-1].open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    aggregate = next(
        row for row in rows
        if int(float(row["video"])) == -1 and int(float(row["user"])) == -1
    )
    with (directory / "latency.json").open(encoding="utf-8") as stream:
        latency = json.load(stream)
    return {
        "mae": float(aggregate["mae"]),
        "rmse": float(aggregate["rmse"]),
        "latency_mean_ms": float(latency["mean_s"]) * 1000.0,
    }


def run_case(command: list[str], directory: Path, resume: bool) -> dict[str, float]:
    metrics_path = directory / "metrics.json"
    if resume and metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            check=True,
        )
    metrics = aggregate_result(directory)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parser().parse_args()
    args.search_run_dir = resolve_from_repo(args.search_run_dir)
    args.output_dir = resolve_from_repo(args.output_dir)
    args.nbs_checkpoint = resolve_from_repo(args.nbs_checkpoint)
    args.projector_checkpoint = resolve_from_repo(args.projector_checkpoint)
    finalists = read_finalists(args.search_run_dir)
    compact_checkpoint = args.output_dir / "compact_checkpoint"

    baseline_dir = args.output_dir / "pure_nbs_compact"
    baseline_model = (
        compact_checkpoint
        if (compact_checkpoint / "compact_adapter.pt").is_file()
        else args.nbs_checkpoint
    )
    baseline_command = build_command(
        args, baseline_model, baseline_dir, None,
        None if baseline_model == compact_checkpoint else compact_checkpoint,
    )
    candidate_commands = [
        build_command(
            args, compact_checkpoint,
            args.output_dir / f"finalist_{int(row['candidate']):02d}_compact",
            row,
        )
        for row in finalists
    ]
    if args.dry_run:
        print(json.dumps({
            "search_run_dir": str(args.search_run_dir),
            "output_dir": str(args.output_dir),
            "baseline": baseline_command,
            "finalists": [
                {"configuration": row, "command": command}
                for row, command in zip(finalists, candidate_commands)
            ],
        }, indent=2))
        return

    for required in (
        args.search_run_dir / "best_configuration.json",
        args.nbs_checkpoint / "adapter_model.bin",
        args.nbs_checkpoint / "modules_except_plm.bin",
        args.nbs_checkpoint / "nash_rank_allocator.pt",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"required input absent: {required}")
    projector_file = (
        args.projector_checkpoint / "modules_except_plm.bin"
        if args.projector_checkpoint.is_dir() else args.projector_checkpoint
    )
    if not projector_file.is_file():
        raise FileNotFoundError(f"projector checkpoint absent: {projector_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    baseline_metrics = run_case(baseline_command, baseline_dir, args.resume)
    if not (compact_checkpoint / "compact_adapter.pt").is_file():
        raise FileNotFoundError(
            f"compact baseline did not produce {compact_checkpoint / 'compact_adapter.pt'}"
        )
    rows.append({
        "method": "Pure NBS-v19 compact", "candidate": "baseline",
        "status": "complete", **baseline_metrics,
    })

    for config, command in zip(finalists, candidate_commands):
        directory = args.output_dir / f"finalist_{int(config['candidate']):02d}_compact"
        row: dict[str, object] = {
            "method": "NBS-v19 compact + kinematic selector",
            **config, "status": "failed",
        }
        try:
            row.update(run_case(command, directory, args.resume))
            row["status"] = "complete"
        except Exception as exc:  # Preserve later finalist evaluations.
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        write_rows(args.output_dir / "comparison_results.csv", rows)
        print(
            f"[finalist {config['candidate']}] {row['status']} "
            f"MAE={row.get('mae')} latency={row.get('latency_mean_ms')}",
            flush=True,
        )

    write_rows(args.output_dir / "comparison_results.csv", rows)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps({
            "nbs_inference_mode": "compact",
            "selection_source": str(args.search_run_dir),
            "test_set_used_for_finalist_selection": False,
            "results": rows,
        }, indent=2), encoding="utf-8",
    )
    print(f"Saved: {args.output_dir / 'comparison_results.csv'}")


if __name__ == "__main__":
    main()
