"""Sweep ABR inference modules, combine Pareto representatives, and confirm seeds.

The fixed NBS-v19 checkpoint is never trained or modified.  Seed 1 screens
temporal, intra-timestep token, and repeat-last speculative parameters in
isolation.  Two representatives per module form at most eight combined runs;
three diverse combined finalists are then evaluated with seeds 2 and 3.
"""

import argparse
import csv
import importlib.util
import itertools
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
BEST5_SCRIPT = Path(__file__).with_name("run_nbs_v19_best5_inference.py")
SPEC = importlib.util.spec_from_file_location("abr_best5", BEST5_SCRIPT)
best5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(best5)

DEFAULT_OUTPUT_DIR = best5.RESULTS_ROOT / "nbs_v19_budget1536_module_sweep"
TEMPORAL_COUNTS = (0, 1, 2, 3, 4, 6)
TOKEN_OFFSET_SETS = (
    (0, 7),
    (2, 6, 7),
    (0, 2, 3, 7),
    (0, 2, 3, 4, 7),
    (0, 1, 2, 3, 6, 7),
    (0, 1, 2, 3, 4, 5, 6, 7),
)
SPECULATIVE_STEPS = (1, 2, 3, 4, 5)
SUMMARY_METRICS = (
    "mean_reward", "qoe_raw_mean", "mean_bitrate_mbps",
    "mean_rebuffer_s_per_chunk", "total_rebuffer_s",
    "mean_smoothness_mbps", "inference_latency_mean_ms",
    "inference_latency_p50_ms", "inference_latency_p95_ms",
    "token_reduction_ratio", "temporal_history_reduction_ratio",
    "intra_token_reduction_ratio", "llm_call_reduction_ratio",
    "acceptance_rate", "target_plm_calls",
)


def baseline_spec():
    return {"name": "nbs_compact_only", "module": "baseline"}


def independent_specs():
    specs = [baseline_spec()]
    specs.extend({
        "name": f"temporal_k{count}", "module": "temporal",
        "temporal": True, "event_max_events": count,
    } for count in TEMPORAL_COUNTS)
    specs.extend({
        "name": "token_o" + "_".join(map(str, offsets)),
        "module": "token", "token": True, "token_offsets": offsets,
        "selection_eligible": len(offsets) < 8,
    } for offsets in TOKEN_OFFSET_SETS)
    specs.extend({
        "name": f"spec_repeat_last_k{steps}", "module": "speculative",
        "speculative": True, "speculative_steps": steps,
        "speculative_drafter": "repeat-last",
    } for steps in SPECULATIVE_STEPS)
    return specs


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def representative_specs(specs, rows, module):
    """Return quality and efficiency representatives for one module."""
    by_name = {row["experiment"]: row for row in rows}
    candidates = [
        spec for spec in specs
        if spec.get("module") == module
        and spec.get("selection_eligible", True)
        and spec["name"] in by_name
        and finite_number(by_name[spec["name"]].get("mean_reward")) is not None
        and finite_number(
            by_name[spec["name"]].get("inference_latency_mean_ms")
        ) is not None
    ]
    if not candidates:
        raise RuntimeError(f"no valid {module} sweep candidates")
    quality = max(
        candidates,
        key=lambda spec: finite_number(by_name[spec["name"]]["mean_reward"]),
    )
    if module == "speculative":
        efficiency_key = lambda spec: (
            finite_number(by_name[spec["name"]].get("llm_call_reduction_ratio"))
            or 0.0,
            -finite_number(by_name[spec["name"]]["inference_latency_mean_ms"]),
        )
        efficiency = max(candidates, key=efficiency_key)
    else:
        efficiency = min(
            candidates,
            key=lambda spec: finite_number(
                by_name[spec["name"]]["inference_latency_mean_ms"]
            ),
        )
    selected = [quality]
    if efficiency["name"] != quality["name"]:
        selected.append(efficiency)
    if len(selected) == 1 and len(candidates) > 1:
        remaining = [spec for spec in candidates if spec != quality]
        selected.append(max(
            remaining,
            key=lambda spec: finite_number(by_name[spec["name"]]["mean_reward"]),
        ))
    return selected[:2]


def combined_specs(selected):
    results = []
    seen = set()
    for temporal, token, speculative in itertools.product(
        selected["temporal"], selected["token"], selected["speculative"]
    ):
        name = f"combined__{temporal['name']}__{token['name']}__{speculative['name']}"
        if name in seen:
            continue
        seen.add(name)
        results.append({
            "name": name, "module": "combined", "temporal": True,
            "event_max_events": temporal["event_max_events"], "token": True,
            "token_offsets": token["token_offsets"], "speculative": True,
            "speculative_steps": speculative["speculative_steps"],
            "speculative_drafter": "repeat-last",
        })
    return results


def combined_finalists(specs, rows, baseline_reward, count=3):
    """Choose quality, acceptable-latency, and balanced finalists."""
    by_name = {row["experiment"]: row for row in rows}
    valid = [
        spec for spec in specs if spec["name"] in by_name
        and finite_number(by_name[spec["name"]].get("mean_reward")) is not None
        and finite_number(
            by_name[spec["name"]].get("inference_latency_mean_ms")
        ) is not None
    ]
    if not valid:
        raise RuntimeError("no valid combined candidates")
    chosen = []

    def add(spec):
        if spec not in chosen and len(chosen) < min(count, len(valid)):
            chosen.append(spec)

    add(max(valid, key=lambda s: finite_number(by_name[s["name"]]["mean_reward"])))
    acceptable = [
        spec for spec in valid
        if finite_number(by_name[spec["name"]]["mean_reward"])
        >= 0.95 * baseline_reward
    ] or valid
    add(min(
        acceptable,
        key=lambda s: finite_number(by_name[s["name"]]["inference_latency_mean_ms"]),
    ))
    rewards = [finite_number(by_name[s["name"]]["mean_reward"]) for s in valid]
    latencies = [
        finite_number(by_name[s["name"]]["inference_latency_mean_ms"])
        for s in valid
    ]
    reward_span = max(rewards) - min(rewards) or 1.0
    latency_span = max(latencies) - min(latencies) or 1.0
    ranked = sorted(valid, key=lambda s: (
        (finite_number(by_name[s["name"]]["mean_reward"]) - min(rewards))
        / reward_span
        + (max(latencies) - finite_number(
            by_name[s["name"]]["inference_latency_mean_ms"]
        )) / latency_span
    ), reverse=True)
    for spec in ranked:
        add(spec)
    return chosen


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_rows(path, rows):
    write_csv(path, rows)
    path.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_rows(path):
    json_path = path.with_suffix(".json")
    if not json_path.is_file():
        return []
    return json.loads(json_path.read_text(encoding="utf-8"))


def run_signature(args):
    return {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "base_model_dir": str(args.base_model_dir.resolve()),
        "exp_pool_path": str(args.exp_pool_path.resolve()),
        "rank_budget": args.rank_budget,
        "physical_rank": args.physical_rank,
        "rank_config": str(args.rank_config),
        "trace": args.trace,
        "trace_num": args.trace_num,
        "video": args.video,
        "confirmation_seeds": list(dict.fromkeys(args.confirmation_seeds)),
        "combined_finalists": args.combined_finalists,
        "temporal_counts": list(TEMPORAL_COUNTS),
        "token_offset_sets": [list(values) for values in TOKEN_OFFSET_SETS],
        "speculative_steps": list(SPECULATIVE_STEPS),
    }


def validate_or_write_manifest(args):
    path = args.output_dir / "pipeline_manifest.json"
    signature = run_signature(args)
    if path.is_file() and args.resume:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise ValueError(
                "existing pipeline manifest does not match this run; use a "
                "different --output-dir or remove --resume"
            )
    elif path.exists() and not args.resume:
        raise FileExistsError(
            f"output already exists: {path}; use --resume or a new --output-dir"
        )
    path.write_text(
        json.dumps({"signature": signature}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def configured_fields(spec):
    return {
        "configured_temporal": bool(spec.get("temporal")),
        "configured_event_max_events": spec.get("event_max_events"),
        "configured_token": bool(spec.get("token")),
        "configured_token_offsets": (
            ",".join(map(str, spec["token_offsets"]))
            if spec.get("token") else None
        ),
        "configured_speculative": bool(spec.get("speculative")),
        "configured_speculative_steps": spec.get("speculative_steps", 0),
        "configured_speculative_drafter": (
            spec.get("speculative_drafter") if spec.get("speculative") else None
        ),
    }


def run_specs(args, specs, seed, output, rows=None):
    rows = list(rows or [])
    completed = {
        (row["experiment"], int(row["data_seed"])) for row in rows
    }
    for spec in specs:
        key = (spec["name"], seed)
        if key in completed:
            print(f"[{seed}:{spec['name']}] already complete; skipping", flush=True)
            continue
        command_args = argparse.Namespace(**vars(args))
        command_args.data_seed = seed
        command = best5.build_command(command_args, spec)
        print(f"[{seed}:{spec['name']}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        started_at = time.time() - 1.0
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        metrics_path = best5.newest_metrics(started_at)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "experiment": spec["name"], "module": spec["module"],
            "data_seed": seed, "rank_budget": args.rank_budget,
            "physical_rank": args.physical_rank,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            **configured_fields(spec), **best5.scalar_metrics(metrics),
        })
        write_rows(output, rows)
    return rows


def aggregate_confirmation(rows, finalist_names):
    summaries = []
    for name in ["nbs_compact_only", *finalist_names]:
        group = [row for row in rows if row["experiment"] == name]
        summary = {
            "experiment": name, "num_seeds": len(group),
            "data_seeds": ",".join(str(row["data_seed"]) for row in group),
        }
        for metric in SUMMARY_METRICS:
            values = [finite_number(row.get(metric)) for row in group]
            if values and all(value is not None for value in values):
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        summaries.append(summary)
    baseline = summaries[0]
    for row in summaries:
        row["mean_reward_change_ratio_vs_nbs"] = (
            row["mean_reward_mean"] / baseline["mean_reward_mean"] - 1.0
        )
        row["inference_latency_reduction_vs_nbs"] = (
            1.0 - row["inference_latency_mean_ms_mean"]
            / baseline["inference_latency_mean_ms_mean"]
        )
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, default=best5.DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path, default=best5.DEFAULT_EXP_POOL)
    parser.add_argument("--rank-budget", type=int, default=1536)
    parser.add_argument("--physical-rank", type=int, default=32)
    parser.add_argument("--rank-config", type=Path,
                        default=Path("configs/nbs_v19_rank_config.json"))
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confirmation-seeds", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--combined-finalists", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run:
        best5.validate_checkpoint(args.checkpoint_dir, args.rank_budget)
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        validate_or_write_manifest(args)
    independent_path = args.output_dir / "independent_seed1.csv"
    independent = run_specs(
        args, independent_specs(), 1, independent_path,
        load_rows(independent_path) if args.resume else [],
    )
    if args.dry_run:
        print("Combined runs are selected after independent metrics exist.")
        return

    selected = {
        module: representative_specs(independent_specs(), independent, module)
        for module in ("temporal", "token", "speculative")
    }
    combined = combined_specs(selected)
    selection_path = args.output_dir / "selected_candidates.json"
    selection_path.write_text(json.dumps({
        "independent": {
            key: [spec["name"] for spec in value] for key, value in selected.items()
        },
        "combined": [spec["name"] for spec in combined],
    }, indent=2, sort_keys=True), encoding="utf-8")

    combined_path = args.output_dir / "combined_seed1.csv"
    combined_rows = run_specs(
        args, combined, 1, combined_path,
        load_rows(combined_path) if args.resume else [],
    )
    baseline_reward = finite_number(next(
        row["mean_reward"] for row in independent
        if row["experiment"] == "nbs_compact_only"
    ))
    finalists = combined_finalists(
        combined, combined_rows, baseline_reward, args.combined_finalists
    )
    existing_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    existing_selection["finalists"] = [spec["name"] for spec in finalists]
    selection_path.write_text(
        json.dumps(existing_selection, indent=2, sort_keys=True), encoding="utf-8"
    )

    confirmation_path = args.output_dir / "confirmation_runs.csv"
    confirmation = load_rows(confirmation_path) if args.resume else []
    seed1_baseline = next(
        row for row in independent if row["experiment"] == "nbs_compact_only"
    )
    if not any(
        row["experiment"] == "nbs_compact_only" and int(row["data_seed"]) == 1
        for row in confirmation
    ):
        confirmation.append(seed1_baseline)
    finalist_names = {spec["name"] for spec in finalists}
    for row in combined_rows:
        if row["experiment"] in finalist_names and not any(
            old["experiment"] == row["experiment"]
            and int(old["data_seed"]) == 1 for old in confirmation
        ):
            confirmation.append(row)
    write_rows(confirmation_path, confirmation)
    for seed in list(dict.fromkeys(args.confirmation_seeds)):
        confirmation = run_specs(
            args, [baseline_spec(), *finalists], seed,
            confirmation_path, confirmation,
        )
    summaries = aggregate_confirmation(
        confirmation, [spec["name"] for spec in finalists]
    )
    summary_path = args.output_dir / "confirmation_summary.csv"
    write_rows(summary_path, summaries)
    print(f"Independent results: {independent_path.resolve()}")
    print(f"Combined results: {combined_path.resolve()}")
    print(f"3-seed summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
