"""Evaluate two focused follow-up groups for the VP cached patch selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.evaluate_cached_patch_selectors_compact import (
    absolute,
    common_command,
    plot_rows,
    run_case,
    validate_inputs,
    write_rows,
)


QUALITY_CASES = (
    ("q_threshold6", {
        "policy": "adaptive", "threshold": 6.0,
        "history": 1, "acceleration": 0.0,
    }),
    ("q_threshold18", {
        "policy": "adaptive", "threshold": 18.0,
        "history": 1, "acceleration": 0.0,
    }),
    ("q_history3_accel05", {
        "policy": "adaptive", "threshold": 12.0,
        "history": 3, "acceleration": 0.5,
    }),
    ("q_cross_history3", {
        "policy": "cross", "threshold": 12.0,
        "history": 3, "acceleration": 0.0,
    }),
)


SPEED_CASES = (
    ("s_gated3_skip1", {
        "policy": "gated-k1", "threshold": 3.0, "max_skip": 1,
    }),
    ("s_gated6_skip2_cache", {
        "policy": "gated-k1", "threshold": 6.0, "max_skip": 2,
        "projector_cache": True,
    }),
    ("s_refresh4_cache", {
        "policy": "adaptive", "threshold": 12.0, "refresh": 4,
        "projector_cache": True,
    }),
    ("s_gated_adaptive6_skip4_cache", {
        "policy": "gated-adaptive", "threshold": 6.0,
        "rapid_multiplier": 2.0, "max_skip": 4,
        "projector_cache": True,
    }),
)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--group", choices=("quality", "speed"), required=True)
    result.add_argument("--nbs-checkpoint", type=Path, required=True)
    result.add_argument("--projector-checkpoint", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--rank-budget", type=int, default=512)
    result.add_argument("--physical-rank", type=int, default=32)
    result.add_argument("--latency-warmup-steps", type=int, default=5)
    result.add_argument("--projector-cache-max-entries", type=int, default=512)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def selector_command(args, model_path, result_dir, config):
    command = common_command(args, model_path, result_dir)
    command.extend([
        "--multimodal-mode", "cached-patch-selection",
        "--cached-patch-policy", config["policy"],
        "--cached-patch-motion-threshold-deg", str(config["threshold"]),
        "--cached-patch-motion-history-window", str(config.get("history", 1)),
        "--cached-patch-acceleration-weight",
        str(config.get("acceleration", 0.0)),
        "--cached-patch-rapid-motion-multiplier",
        str(config.get("rapid_multiplier", 2.0)),
        "--cached-patch-refresh-interval", str(config.get("refresh", 1)),
        "--cached-patch-max-skip-calls", str(config.get("max_skip", 0)),
        "--cached-patch-projector-cache-max-entries",
        str(args.projector_cache_max_entries),
        "--cached-patch-features-dir", str(args.cache_dir),
        "--cached-patch-cache-device", "model",
        "--cached-patch-preload",
        "--cached-patch-stats-output-path",
        str(result_dir / "selector_stats.json"),
        "--multimodal-projector-checkpoint", str(args.projector_checkpoint),
        "--inference-tag", result_dir.name,
    ])
    if config.get("projector_cache"):
        command.append("--cached-patch-projector-cache")
    return command


def main():
    args = parser().parse_args()
    for name in ("nbs_checkpoint", "projector_checkpoint", "cache_dir",
                 "output_dir"):
        setattr(args, name, absolute(getattr(args, name)))
    if args.projector_cache_max_entries <= 0:
        raise ValueError("--projector-cache-max-entries must be positive")
    cases = QUALITY_CASES if args.group == "quality" else SPEED_CASES
    compact_dir = args.output_dir / "compact_checkpoint"
    compact_exists = (compact_dir / "compact_adapter.pt").is_file()
    baseline_model = compact_dir if compact_exists else args.nbs_checkpoint
    commands = [("pure_nbs_compact", None, common_command(
        args, baseline_model, args.output_dir / "pure_nbs_compact"
    ))]
    if not compact_exists:
        commands[0][2].extend(["--nbs-compact-output-dir", str(compact_dir)])
    commands.extend(
        (name, config, selector_command(
            args, compact_dir, args.output_dir / name, config
        ))
        for name, config in cases
    )
    if args.dry_run:
        print(json.dumps([
            {"case": name, "config": config, "command": command}
            for name, config, command in commands
        ], indent=2))
        return

    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, config, command in commands:
        row = {
            "case": name,
            "group": args.group,
            "status": "failed",
            **(config or {}),
        }
        try:
            row.update(run_case(
                command, args.output_dir / name, args.resume
            ))
            row["status"] = "complete"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        write_rows(args.output_dir / "comparison_results.csv", rows)
        print(
            f"[{name}] {row['status']} MAE={row.get('mae')} "
            f"latency={row.get('latency_mean_ms')}", flush=True,
        )
        if (name == "pure_nbs_compact"
                and not (compact_dir / "compact_adapter.pt").is_file()):
            raise RuntimeError("baseline did not produce compact checkpoint")

    completed = [row for row in rows if row["status"] == "complete"]
    if completed:
        plot_rows(
            completed,
            args.output_dir / f"cached_patch_{args.group}_overview.png",
        )
    print(f"Saved: {args.output_dir / 'comparison_results.csv'}")


if __name__ == "__main__":
    main()
