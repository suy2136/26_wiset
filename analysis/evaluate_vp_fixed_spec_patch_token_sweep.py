"""Sweep VP patch/token settings with speculative G=6, threshold=0.4 fixed.

The compact checkpoint, pure-NBS baseline, and speculative-only reference are
reused from a completed VP module run.  Stage 1 evaluates Spec+Token for
Recent-K 6/8/10.  Stage 2 selects the fastest K within the configured NBS MAE
constraint and evaluates eight conservative cached-patch configurations.
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

from analysis.evaluate_cached_patch_followups_compact import selector_command
from analysis.evaluate_cached_patch_selectors_compact import (
    absolute,
    common_command,
    plot_rows,
    run_case,
    write_rows,
)
from analysis.evaluate_vp_inference_module_pipeline import (
    finite,
    set_option,
    trace_metrics,
    trace_option,
    write_json,
)


SPEC_GAMMA = 6
SPEC_THRESHOLD = 0.4
TOKEN_K_VALUES = (6, 8, 10)
PATCH_CASES = (
    ("gated_k1_t3_skip0_cache", {
        "policy": "gated-k1", "threshold": 3.0,
        "max_skip": 0, "projector_cache": True,
    }),
    ("gated_k1_t3_skip1_cache", {
        "policy": "gated-k1", "threshold": 3.0,
        "max_skip": 1, "projector_cache": True,
    }),
    ("gated_k1_t6_skip0_cache", {
        "policy": "gated-k1", "threshold": 6.0,
        "max_skip": 0, "projector_cache": True,
    }),
    ("gated_k1_t6_skip1_cache", {
        "policy": "gated-k1", "threshold": 6.0,
        "max_skip": 1, "projector_cache": True,
    }),
    ("gated_adaptive_t3_skip1_cache", {
        "policy": "gated-adaptive", "threshold": 3.0,
        "rapid_multiplier": 2.0, "max_skip": 1,
        "projector_cache": True,
    }),
    ("gated_adaptive_t3_skip2_cache", {
        "policy": "gated-adaptive", "threshold": 3.0,
        "rapid_multiplier": 2.0, "max_skip": 2,
        "projector_cache": True,
    }),
    ("gated_adaptive_t6_skip1_cache", {
        "policy": "gated-adaptive", "threshold": 6.0,
        "rapid_multiplier": 2.0, "max_skip": 1,
        "projector_cache": True,
    }),
    ("gated_adaptive_t6_skip2_cache", {
        "policy": "gated-adaptive", "threshold": 6.0,
        "rapid_multiplier": 2.0, "max_skip": 2,
        "projector_cache": True,
    }),
)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--reference-run-dir", type=Path, required=True)
    result.add_argument("--projector-checkpoint", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--rank-budget", type=int, default=512)
    result.add_argument("--physical-rank", type=int, default=32)
    result.add_argument("--latency-warmup-steps", type=int, default=5)
    result.add_argument("--projector-cache-max-entries", type=int, default=512)
    result.add_argument("--max-mae-increase-ratio", type=float, default=0.03)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def load_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def reference_row(rows, case):
    matches = [
        row for row in rows
        if row.get("case") == case and row.get("status") == "complete"
        and finite(row.get("mae")) is not None
        and finite(row.get("latency_mean_ms")) is not None
    ]
    if len(matches) != 1:
        raise ValueError(f"reference run must contain one complete {case}")
    row = dict(matches[0])
    for key in ("mae", "rmse", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms"):
        if finite(row.get(key)) is not None:
            row[key] = finite(row[key])
    row["source"] = "reference_run"
    return row


def validate_inputs(args):
    compact = args.reference_run_dir / "compact_checkpoint"
    for name in ("compact_adapter.pt", "modules_except_plm.bin"):
        if not (compact / name).is_file():
            raise FileNotFoundError(f"compact checkpoint input absent: {compact / name}")
    projector = (
        args.projector_checkpoint / "modules_except_plm.bin"
        if args.projector_checkpoint.is_dir() else args.projector_checkpoint
    )
    if not projector.is_file():
        raise FileNotFoundError(f"projector checkpoint absent: {projector}")
    for video in (4, 8, 14, 18, 24, 25):
        path = args.cache_dir / f"video{video}_patch_features.pt"
        if not path.is_file():
            raise FileNotFoundError(f"test-video patch cache absent: {path}")
    source = args.reference_run_dir / "vp_module_sweep_results.csv"
    if not source.is_file():
        raise FileNotFoundError(f"reference results absent: {source}")
    references = load_rows(source)
    return (
        compact,
        reference_row(references, "pure_nbs_compact"),
        reference_row(references, "spec_g6_t0p4"),
    )


def spec_token_command(args, compact, result_dir, recent_k):
    command = common_command(args, compact, result_dir)
    command.extend([
        "--selector-recent-k", str(recent_k),
        "--speculative-gamma", str(SPEC_GAMMA),
        "--speculative-threshold", str(SPEC_THRESHOLD),
        "--inference-tag", "full_stack",
        *trace_option(result_dir),
    ])
    return command


def full_stack_command(args, compact, result_dir, patch_config, recent_k):
    command = selector_command(args, compact, result_dir, patch_config)
    set_option(command, "--inference-tag", "full_stack")
    command.extend([
        "--selector-recent-k", str(recent_k),
        "--speculative-gamma", str(SPEC_GAMMA),
        "--speculative-threshold", str(SPEC_THRESHOLD),
        *trace_option(result_dir),
    ])
    return command


def run_one(args, case, family, config, command):
    result_dir = args.output_dir / case
    row = {"case": case, "family": family, "status": "failed", **config}
    if args.dry_run:
        row["command"] = command
        print(f"[{case}] dry-run", flush=True)
        return row
    try:
        row.update(run_case(command, result_dir, args.resume))
        row.update(trace_metrics(result_dir))
        row["status"] = "complete"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    print(
        f"[{case}] {row['status']} MAE={row.get('mae')} "
        f"latency={row.get('latency_mean_ms')}", flush=True,
    )
    return row


def select_fastest(rows, baseline_mae, max_ratio):
    candidates = [
        row for row in rows
        if row.get("family") == "spec_token" and row.get("status") == "complete"
        and finite(row.get("mae")) is not None
        and finite(row.get("latency_mean_ms")) is not None
    ]
    if not candidates:
        raise RuntimeError("no complete Spec+Token candidates")
    admissible = [
        row for row in candidates
        if finite(row["mae"]) <= baseline_mae * (1.0 + max_ratio)
    ]
    if not admissible:
        return min(candidates, key=lambda row: finite(row["mae"]))
    return min(admissible, key=lambda row: finite(row["latency_mean_ms"]))


def select_finalists(rows, baseline_mae, max_ratio):
    candidates = [
        row for row in rows
        if row.get("family") == "full_stack" and row.get("status") == "complete"
    ]
    if not candidates:
        raise RuntimeError("no complete full-stack candidates")
    quality = min(candidates, key=lambda row: finite(row["mae"]))
    admissible = [
        row for row in candidates
        if finite(row["mae"]) <= baseline_mae * (1.0 + max_ratio)
    ]
    speed = min(
        admissible or candidates,
        key=lambda row: finite(row["latency_mean_ms"]),
    )
    return quality, speed


def write_pipeline_rows(args, rows):
    path = args.output_dir / "fixed_spec_patch_token_sweep.csv"
    write_rows(path, rows)
    write_json(path.with_suffix(".json"), rows)


def write_manifest(args):
    path = args.output_dir / "fixed_spec_patch_token_manifest.json"
    signature = {
        "reference_run_dir": str(args.reference_run_dir),
        "projector_checkpoint": str(args.projector_checkpoint),
        "cache_dir": str(args.cache_dir),
        "rank_budget": args.rank_budget,
        "physical_rank": args.physical_rank,
        "spec_gamma": SPEC_GAMMA,
        "spec_threshold": SPEC_THRESHOLD,
        "token_k_values": list(TOKEN_K_VALUES),
        "patch_cases": [{"name": name, **config} for name, config in PATCH_CASES],
        "max_mae_increase_ratio": args.max_mae_increase_ratio,
    }
    if path.is_file() and args.resume:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError("resume manifest differs from current configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    write_json(path, {"signature": signature})


def comparison_rows(baseline, spec_only, token_best, finalists):
    output, seen = [], set()
    for row in (baseline, spec_only, token_best, *finalists):
        if row["case"] in seen:
            continue
        seen.add(row["case"])
        item = dict(row)
        item["mae_change_percent_vs_nbs"] = (
            finite(item["mae"]) / finite(baseline["mae"]) - 1.0
        ) * 100.0
        item["latency_reduction_percent_vs_nbs"] = (
            1.0 - finite(item["latency_mean_ms"])
            / finite(baseline["latency_mean_ms"])
        ) * 100.0
        output.append(item)
    return output


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("reference_run_dir", "projector_checkpoint", "cache_dir", "output_dir"):
        setattr(args, name, absolute(getattr(args, name)))
    if not 0 <= args.max_mae_increase_ratio < 1:
        raise ValueError("--max-mae-increase-ratio must be in [0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        compact = args.reference_run_dir / "compact_checkpoint"
        baseline = {
            "case": "pure_nbs_compact", "family": "baseline",
            "status": "complete", "mae": 1.0, "latency_mean_ms": 1.0,
        }
        spec_only = {
            "case": "spec_g6_t0p4", "family": "speculative",
            "status": "complete", "mae": 1.0, "latency_mean_ms": 1.0,
        }
    else:
        compact, baseline, spec_only = validate_inputs(args)
        write_manifest(args)
    rows = [baseline, spec_only]

    for recent_k in TOKEN_K_VALUES:
        case = f"spec_g6_t0p4_token_k{recent_k}"
        result_dir = args.output_dir / case
        rows.append(run_one(
            args, case, "spec_token", {"recent_k": recent_k},
            spec_token_command(args, compact, result_dir, recent_k),
        ))
        if not args.dry_run:
            write_pipeline_rows(args, rows)
    if args.dry_run:
        selected_k = 8
    else:
        selected = select_fastest(
            rows, finite(baseline["mae"]), args.max_mae_increase_ratio
        )
        selected_k = int(selected["recent_k"])
        write_json(args.output_dir / "selected_token_setting.json", selected)
        print(f"Selected token K={selected_k}", flush=True)

    for name, patch_config in PATCH_CASES:
        case = f"full_{name}_token_k{selected_k}_spec_g6_t0p4"
        result_dir = args.output_dir / case
        rows.append(run_one(
            args, case, "full_stack",
            {"patch_case": name, "recent_k": selected_k,
             "gamma": SPEC_GAMMA, "acceptance_threshold": SPEC_THRESHOLD,
             **patch_config},
            full_stack_command(
                args, compact, result_dir, patch_config, selected_k
            ),
        ))
        if not args.dry_run:
            write_pipeline_rows(args, rows)
    if args.dry_run:
        print("Dry run: 3 Spec+Token + 8 full-stack evaluations", flush=True)
        return

    failed = [row["case"] for row in rows if row.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"failed cases: {', '.join(failed)}")
    quality, speed = select_finalists(
        rows, finite(baseline["mae"]), args.max_mae_increase_ratio
    )
    final = comparison_rows(baseline, spec_only, selected, (quality, speed))
    final_path = args.output_dir / "fixed_spec_final_comparison.csv"
    write_rows(final_path, final)
    write_json(final_path.with_suffix(".json"), final)
    plot_rows(final, args.output_dir / "fixed_spec_final_comparison.png")
    print(f"Sweep results: {args.output_dir / 'fixed_spec_patch_token_sweep.csv'}")
    print(f"Final comparison: {final_path}")


if __name__ == "__main__":
    main()
