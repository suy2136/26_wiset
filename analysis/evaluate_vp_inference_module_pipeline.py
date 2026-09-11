"""Long VP inference sweep over cached patches, tokens, and speculation.

The pipeline reuses an existing pure-NBS compact result/checkpoint when the
same output directory is supplied with ``--resume``.  It evaluates cached
patch policies, Recent-K token selection, and speculative decoding in
isolation, selects accuracy/latency representatives, and only then runs the
two composed full-stack finalists.  Every child process streams to both the
terminal and its ``run.log`` through the shared cached-selector runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.evaluate_cached_patch_followups_compact import (
    QUALITY_CASES,
    SPEED_CASES,
    selector_command,
)
from analysis.evaluate_cached_patch_selectors_compact import (
    absolute,
    common_command,
    plot_rows,
    run_case,
    validate_inputs,
    write_rows,
)


ADDITIONAL_PATCH_CASES = (
    ("p_fixed_k1", {"policy": "k1", "threshold": 12.0}),
    ("p_fixed_k2", {"policy": "k2", "threshold": 12.0}),
    ("p_adaptive_t9", {"policy": "adaptive", "threshold": 9.0}),
    ("p_adaptive_t12", {"policy": "adaptive", "threshold": 12.0}),
    ("p_adaptive_t15", {"policy": "adaptive", "threshold": 15.0}),
    ("p_adaptive_t12_h2_a025", {
        "policy": "adaptive", "threshold": 12.0,
        "history": 2, "acceleration": 0.25,
    }),
    ("p_refresh2_cache", {
        "policy": "adaptive", "threshold": 12.0,
        "refresh": 2, "projector_cache": True,
    }),
    ("p_gated3_skip2_cache", {
        "policy": "gated-k1", "threshold": 3.0,
        "max_skip": 2, "projector_cache": True,
    }),
)
PATCH_CASES = QUALITY_CASES + SPEED_CASES + ADDITIONAL_PATCH_CASES
TOKEN_K_VALUES = (2, 4, 6, 8)
SPECULATIVE_CONFIGS = tuple(
    (gamma, threshold)
    for gamma in (2, 4, 6)
    for threshold in (0.2, 0.3, 0.4)
)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--nbs-checkpoint", type=Path, required=True)
    result.add_argument("--projector-checkpoint", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--rank-budget", type=int, default=512)
    result.add_argument("--physical-rank", type=int, default=32)
    result.add_argument("--latency-warmup-steps", type=int, default=5)
    result.add_argument("--projector-cache-max-entries", type=int, default=512)
    result.add_argument(
        "--max-mae-increase-ratio", type=float, default=0.03,
        help="Maximum MAE increase allowed when choosing latency candidates.",
    )
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def option(command, name):
    return command[command.index(name) + 1]


def set_option(command, name, value):
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend([name, str(value)])


def trace_option(result_dir):
    return [
        "--inference-trace-output-path", str(result_dir / "inference_trace.json")
    ]


def token_command(args, compact_dir, result_dir, recent_k):
    command = common_command(args, compact_dir, result_dir)
    command.extend([
        "--selector-recent-k", str(recent_k),
        "--inference-tag", "selector",
        *trace_option(result_dir),
    ])
    return command


def speculative_command(args, compact_dir, result_dir, gamma, threshold):
    command = common_command(args, compact_dir, result_dir)
    command.extend([
        "--speculative-gamma", str(gamma),
        "--speculative-threshold", str(threshold),
        "--inference-tag", "speculative",
        *trace_option(result_dir),
    ])
    return command


def full_stack_command(
    args, compact_dir, result_dir, patch_config, recent_k, gamma, threshold,
):
    command = selector_command(args, compact_dir, result_dir, patch_config)
    set_option(command, "--inference-tag", "full_stack")
    command.extend([
        "--selector-recent-k", str(recent_k),
        "--speculative-gamma", str(gamma),
        "--speculative-threshold", str(threshold),
        *trace_option(result_dir),
    ])
    return command


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def trace_metrics(result_dir):
    path = result_dir / "inference_trace.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: value for key, value in payload.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def run_one(args, name, family, config, command):
    result_dir = args.output_dir / name
    row = {"case": name, "family": family, "status": "failed", **config}
    if args.dry_run:
        row["command"] = command
        return row
    try:
        row.update(run_case(command, result_dir, args.resume))
        row.update(trace_metrics(result_dir))
        row["status"] = "complete"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    print(
        f"[{name}] {row['status']} MAE={row.get('mae')} "
        f"latency={row.get('latency_mean_ms')}", flush=True,
    )
    return row


def valid_family(rows, family):
    return [
        row for row in rows
        if row.get("family") == family and row.get("status") == "complete"
        and finite(row.get("mae")) is not None
        and finite(row.get("latency_mean_ms")) is not None
    ]


def choose_accuracy_and_speed(rows, family, baseline_mae, max_ratio):
    candidates = valid_family(rows, family)
    if not candidates:
        raise RuntimeError(f"no complete {family} candidates")
    accuracy = min(candidates, key=lambda row: finite(row["mae"]))
    admissible = [
        row for row in candidates
        if finite(row["mae"]) <= baseline_mae * (1.0 + max_ratio)
    ]
    speed = min(
        admissible or candidates,
        key=lambda row: finite(row["latency_mean_ms"]),
    )
    return accuracy, speed


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_manifest(args):
    path = args.output_dir / "vp_module_pipeline_manifest.json"
    signature = {
        "nbs_checkpoint": str(args.nbs_checkpoint),
        "projector_checkpoint": str(args.projector_checkpoint),
        "cache_dir": str(args.cache_dir),
        "rank_budget": args.rank_budget,
        "physical_rank": args.physical_rank,
        "max_mae_increase_ratio": args.max_mae_increase_ratio,
        "patch_cases": [name for name, _ in PATCH_CASES],
        "token_k_values": list(TOKEN_K_VALUES),
        "speculative_configs": [list(values) for values in SPECULATIVE_CONFIGS],
    }
    if path.is_file() and args.resume:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError(
                "existing pipeline manifest differs; choose another output "
                "directory or remove --resume"
            )
    elif path.exists() and not args.resume:
        raise FileExistsError(f"pipeline output already exists: {path}")
    write_json(path, {"signature": signature})


def write_pipeline_rows(args, rows):
    write_rows(args.output_dir / "vp_module_sweep_results.csv", rows)
    write_json(args.output_dir / "vp_module_sweep_results.json", rows)


def config_from_patch_row(row):
    return dict(PATCH_CASES)[row["case"]]


def final_comparison_rows(baseline, patch_pair, token_best, spec_best, stacks):
    ordered = [baseline, token_best, spec_best]
    for row in patch_pair:
        if row["case"] not in {item["case"] for item in ordered}:
            ordered.append(row)
    ordered.extend(stacks)
    baseline_mae = finite(baseline["mae"])
    baseline_latency = finite(baseline["latency_mean_ms"])
    output = []
    for row in ordered:
        item = dict(row)
        item["mae_change_percent_vs_nbs"] = (
            (finite(row["mae"]) / baseline_mae - 1.0) * 100.0
        )
        item["latency_reduction_percent_vs_nbs"] = (
            (1.0 - finite(row["latency_mean_ms"]) / baseline_latency) * 100.0
        )
        output.append(item)
    return output


def plot_final(rows, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped final PNG", flush=True)
        return
    labels = [row["case"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, len(rows) * 0.55)))
    axes[0].barh(labels, [row["mae"] for row in rows], color="#3977B7")
    axes[0].invert_yaxis()
    axes[0].set_title("VP MAE (lower is better)")
    axes[0].set_xlabel("MAE (degrees)")
    axes[1].barh(
        labels, [row["latency_mean_ms"] for row in rows], color="#4E9F6D"
    )
    axes[1].invert_yaxis()
    axes[1].set_title("Mean inference latency (lower is better)")
    axes[1].set_xlabel("Milliseconds")
    fig.suptitle("NBS-v19 compact: VP inference-module finalists")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("nbs_checkpoint", "projector_checkpoint", "cache_dir", "output_dir"):
        setattr(args, name, absolute(getattr(args, name)))
    if not 0 <= args.max_mae_increase_ratio < 1:
        raise ValueError("--max-mae-increase-ratio must be in [0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        validate_inputs(args)
        write_manifest(args)

    compact_dir = args.output_dir / "compact_checkpoint"
    compact_exists = (compact_dir / "compact_adapter.pt").is_file()
    baseline_dir = args.output_dir / "pure_nbs_compact"
    baseline_command = common_command(
        args, compact_dir if compact_exists else args.nbs_checkpoint, baseline_dir
    )
    if not compact_exists:
        baseline_command.extend(["--nbs-compact-output-dir", str(compact_dir)])

    rows = []
    baseline = run_one(args, "pure_nbs_compact", "baseline", {}, baseline_command)
    rows.append(baseline)
    if args.dry_run:
        baseline_mae = 1.0
    else:
        if baseline["status"] != "complete":
            raise RuntimeError("pure NBS compact baseline failed")
        if not (compact_dir / "compact_adapter.pt").is_file():
            raise RuntimeError("compact checkpoint was not produced")
        baseline_mae = finite(baseline["mae"])

    for name, config in PATCH_CASES:
        result_dir = args.output_dir / name
        command = selector_command(args, compact_dir, result_dir, config)
        rows.append(run_one(args, name, "patch", config, command))
        if not args.dry_run:
            write_pipeline_rows(args, rows)

    for recent_k in TOKEN_K_VALUES:
        name = f"token_recent_k{recent_k}"
        result_dir = args.output_dir / name
        rows.append(run_one(
            args, name, "token", {"recent_k": recent_k},
            token_command(args, compact_dir, result_dir, recent_k),
        ))
        if not args.dry_run:
            write_pipeline_rows(args, rows)

    for gamma, threshold in SPECULATIVE_CONFIGS:
        threshold_tag = str(threshold).replace(".", "p")
        name = f"spec_g{gamma}_t{threshold_tag}"
        result_dir = args.output_dir / name
        rows.append(run_one(
            args, name, "speculative",
            {"gamma": gamma, "acceptance_threshold": threshold},
            speculative_command(
                args, compact_dir, result_dir, gamma, threshold
            ),
        ))
        if not args.dry_run:
            write_pipeline_rows(args, rows)

    if args.dry_run:
        print(
            f"Dry run: baseline + {len(PATCH_CASES)} patch + "
            f"{len(TOKEN_K_VALUES)} token + {len(SPECULATIVE_CONFIGS)} "
            "speculative commands; full-stack selection is deferred."
        )
        return

    patch_accuracy, patch_speed = choose_accuracy_and_speed(
        rows, "patch", baseline_mae, args.max_mae_increase_ratio
    )
    _, token_best = choose_accuracy_and_speed(
        rows, "token", baseline_mae, args.max_mae_increase_ratio
    )
    _, spec_best = choose_accuracy_and_speed(
        rows, "speculative", baseline_mae, args.max_mae_increase_ratio
    )
    selected_patches = [patch_accuracy]
    if patch_speed["case"] != patch_accuracy["case"]:
        selected_patches.append(patch_speed)
    selection = {
        "patch_accuracy": patch_accuracy["case"],
        "patch_speed": patch_speed["case"],
        "token_best": token_best["case"],
        "speculative_best": spec_best["case"],
        "mae_constraint_ratio": args.max_mae_increase_ratio,
    }
    write_json(args.output_dir / "selected_module_settings.json", selection)

    stack_rows = []
    for patch_row in selected_patches:
        patch_config = config_from_patch_row(patch_row)
        name = f"full__{patch_row['case']}__{token_best['case']}__{spec_best['case']}"
        result_dir = args.output_dir / name
        config = {
            "patch_case": patch_row["case"],
            "recent_k": int(token_best["recent_k"]),
            "gamma": int(spec_best["gamma"]),
            "acceptance_threshold": float(spec_best["acceptance_threshold"]),
        }
        stack = run_one(
            args, name, "full_stack", config,
            full_stack_command(
                args, compact_dir, result_dir, patch_config,
                config["recent_k"], config["gamma"],
                config["acceptance_threshold"],
            ),
        )
        rows.append(stack)
        stack_rows.append(stack)
        write_pipeline_rows(args, rows)
    if any(row["status"] != "complete" for row in stack_rows):
        raise RuntimeError("one or more full-stack finalists failed")

    final_rows = final_comparison_rows(
        baseline, selected_patches, token_best, spec_best, stack_rows
    )
    final_path = args.output_dir / "vp_module_final_comparison.csv"
    write_rows(final_path, final_rows)
    write_json(final_path.with_suffix(".json"), final_rows)
    plot_final(final_rows, args.output_dir / "vp_module_final_comparison.png")
    print(f"All sweep results: {args.output_dir / 'vp_module_sweep_results.csv'}")
    print(f"Final comparison: {final_path}")


if __name__ == "__main__":
    main()
