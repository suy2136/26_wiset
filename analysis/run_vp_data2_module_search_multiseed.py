"""Search VP inference modules on data/evaluation seed 2 and confirm on 1/2/3.

The supplied compact NBS checkpoint, cached patch features, and projector are
strictly read-only.  Every artifact is written below ``--output-dir``.  All
evaluations use continuous VP RNG and the FP16 fast path with prescaled-Q/K
fallback only for anomalous outputs, matching the fixed VP protocol.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.evaluate_cached_patch_followups_compact import selector_command
from analysis.evaluate_cached_patch_selectors_compact import (
    absolute, common_command, run_case, write_rows,
)
from analysis.evaluate_vp_fast_fallback_fullstack_multiseed import read_safety
from analysis.evaluate_vp_inference_module_pipeline import (
    full_stack_command, set_option, speculative_command, token_command,
    trace_metrics,
)


SEARCH_SEED = 2
CONFIRMATION_SEEDS = (1, 3)
ALL_SEEDS = (1, 2, 3)
TOKEN_K_VALUES = (4, 6, 10)
SPECULATIVE_CONFIGS = (
    (4, 0.3), (4, 0.4), (6, 0.3), (8, 0.3), (8, 0.4),
)
PATCH_CASES = (
    ("patch_gated_k1_t3_skip0_cache", {
        "policy": "gated-k1", "threshold": 3.0, "max_skip": 0,
        "projector_cache": True,
    }),
    ("patch_gated_k1_t6_skip1_cache", {
        "policy": "gated-k1", "threshold": 6.0, "max_skip": 1,
        "projector_cache": True,
    }),
    ("patch_gated_k1_t9_skip1_cache", {
        "policy": "gated-k1", "threshold": 9.0, "max_skip": 1,
        "projector_cache": True,
    }),
)
MAX_FULL_STACK_CANDIDATES = 4
EXCLUDED_SERVER1_CASES = {
    "patch": "gated-k1/T6/skip0/cache",
    "token": "K=8",
    "speculative": "G=6/T=0.4",
    "full_stack": "T6/skip0 + K8 + G6/T0.4",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--compact-checkpoint", type=Path, required=True)
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
        help="Maximum MAE increase versus pure NBS for speed representatives.",
    )
    result.add_argument(
        "--final-quality-slack-ratio", type=float, default=0.01,
        help="Choose the fastest full stack within this ratio of best full-stack MAE.",
    )
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(args) -> dict:
    return {
        "pipeline": "vp_data2_module_search_v2",
        "compact_checkpoint": str(args.compact_checkpoint),
        "projector_checkpoint": str(args.projector_checkpoint),
        "cache_dir": str(args.cache_dir),
        "rank_budget": args.rank_budget,
        "physical_rank": args.physical_rank,
        "search_seed": SEARCH_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "evaluation_rng_mode": "continuous",
        "attention_score_mode": "fp16_fast_with_prescaled_qk_fallback",
        "patch_cases": [{"case": name, **config} for name, config in PATCH_CASES],
        "token_k_values": list(TOKEN_K_VALUES),
        "speculative_configs": [list(item) for item in SPECULATIVE_CONFIGS],
        "excluded_server1_cases": EXCLUDED_SERVER1_CASES,
        "max_full_stack_candidates": MAX_FULL_STACK_CANDIDATES,
        "max_mae_increase_ratio": args.max_mae_increase_ratio,
        "final_quality_slack_ratio": args.final_quality_slack_ratio,
    }


def validate(args) -> None:
    required = (
        args.compact_checkpoint / "compact_adapter.pt",
        args.compact_checkpoint / "modules_except_plm.bin",
        args.compact_checkpoint / "equivalence_report.json",
        args.cache_dir / "manifest.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.projector_checkpoint.is_file():
        raise FileNotFoundError(args.projector_checkpoint)
    patch_files = list(args.cache_dir.glob("video*_patch_features.pt"))
    if len(patch_files) != 27:
        raise ValueError(f"expected 27 cached videos, found {len(patch_files)}")
    report = json.loads(
        (args.compact_checkpoint / "equivalence_report.json").read_text(
            encoding="utf-8"
        )
    )
    if not report.get("passed"):
        raise RuntimeError("compact checkpoint equivalence report did not pass")


def prepare_manifest(args) -> None:
    path = args.output_dir / "pipeline_manifest.json"
    expected = signature(args)
    if path.is_file():
        if not args.resume:
            raise FileExistsError(f"output already exists: {args.output_dir}")
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual.get("signature") != expected:
            raise ValueError("resume configuration differs from pipeline manifest")
    else:
        write_json(path, {"signature": expected})


def fixed_protocol(command: list[str], seed: int) -> list[str]:
    command = [
        item for item in command
        if item not in ("--vp-fp16-fallback", "--vp-fp16-prescaled-qk")
    ]
    command.append("--vp-fp16-fallback")
    set_option(command, "--seed", seed)
    set_option(command, "--lora-seed", 1)
    set_option(command, "--data-seed", seed)
    set_option(command, "--evaluation-rng-mode", "continuous")
    return command


def result_directory(args, case: str, seed: int) -> Path:
    return args.output_dir / "runs" / case / f"seed_{seed}"


def run_candidate(args, rows: list[dict], case: str, family: str,
                  config: dict, seed: int, command: list[str]) -> dict:
    matches = [
        row for row in rows
        if row.get("case") == case
        and int(row.get("evaluation_seed", -1)) == seed
        and row.get("status") == "complete"
    ]
    if matches:
        print(f"[{case} seed={seed}] already complete; skipping", flush=True)
        return matches[-1]
    result_dir = result_directory(args, case, seed)
    command = fixed_protocol(command, seed)
    row = {
        "case": case, "family": family, "evaluation_seed": seed,
        "status": "failed", **config,
    }
    print(f"[{case} seed={seed}] starting", flush=True)
    if args.dry_run:
        row.update({"status": "dry-run", "command": command})
        return row
    try:
        row.update(run_case(command, result_dir, args.resume))
        row.update(trace_metrics(result_dir))
        row.update(read_safety(result_dir))
        row["status"] = "complete"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    rows.append(row)
    write_rows(args.output_dir / "all_runs.csv", rows)
    print(
        f"[{case} seed={seed}] {row['status']} MAE={row.get('mae')} "
        f"latency={row.get('latency_mean_ms')}", flush=True,
    )
    return row


def representatives(rows: list[dict], family: str, baseline_mae: float,
                    max_ratio: float) -> list[dict]:
    candidates = [
        row for row in rows
        if row.get("family") == family
        and int(row.get("evaluation_seed", -1)) == SEARCH_SEED
        and row.get("status") == "complete"
        and finite(row.get("mae")) is not None
        and finite(row.get("latency_mean_ms")) is not None
    ]
    if not candidates:
        raise RuntimeError(f"no complete {family} candidates")
    quality = min(candidates, key=lambda row: finite(row["mae"]))
    admissible = [
        row for row in candidates
        if finite(row["mae"]) <= baseline_mae * (1.0 + max_ratio)
    ]
    speed = min(
        admissible or candidates,
        key=lambda row: finite(row["latency_mean_ms"]),
    )
    return [quality] if quality["case"] == speed["case"] else [quality, speed]


def patch_config(row: dict) -> dict:
    return dict(PATCH_CASES)[row["case"]]


def stack_case(patch: dict, token: dict, speculative: dict) -> str:
    return (
        f"full__{patch['case']}__token_k{int(token['recent_k'])}"
        f"__spec_g{int(speculative['gamma'])}_t"
        f"{str(float(speculative['acceptance_threshold'])).replace('.', 'p')}"
    )


def full_stack_combinations(selected: dict) -> list[tuple[dict, dict, dict]]:
    """Return at most four diverse quality/speed representative combinations."""
    groups = (
        selected["patch"], selected["token"], selected["speculative"],
    )
    combinations = list(itertools.product(*groups))
    if len(combinations) <= MAX_FULL_STACK_CANDIDATES:
        return combinations

    preferred_indices = (
        (0, 0, 0),       # all quality representatives
        (-1, -1, -1),    # all speed representatives
        (0, -1, -1),     # quality patch, faster token/spec
        (-1, 0, 0),      # faster patch, quality token/spec
    )
    chosen = []
    seen = set()
    for indices in preferred_indices:
        candidate = tuple(group[index] for group, index in zip(groups, indices))
        identity = tuple(row["case"] for row in candidate)
        if identity not in seen:
            chosen.append(candidate)
            seen.add(identity)
    for candidate in combinations:
        if len(chosen) >= MAX_FULL_STACK_CANDIDATES:
            break
        identity = tuple(row["case"] for row in candidate)
        if identity not in seen:
            chosen.append(candidate)
            seen.add(identity)
    return chosen[:MAX_FULL_STACK_CANDIDATES]


def choose_final(rows: list[dict], quality_slack: float) -> dict:
    candidates = [
        row for row in rows
        if row.get("family") == "full_stack"
        and int(row.get("evaluation_seed", -1)) == SEARCH_SEED
        and row.get("status") == "complete"
    ]
    if not candidates:
        raise RuntimeError("no complete full-stack candidates")
    best_mae = min(finite(row["mae"]) for row in candidates)
    admissible = [
        row for row in candidates
        if finite(row["mae"]) <= best_mae * (1.0 + quality_slack)
    ]
    return min(admissible, key=lambda row: finite(row["latency_mean_ms"]))


def summarize_final(rows: list[dict], case: str) -> dict:
    group = sorted(
        (
            row for row in rows
            if row.get("case") == case and row.get("status") == "complete"
        ),
        key=lambda row: int(row["evaluation_seed"]),
    )
    if [int(row["evaluation_seed"]) for row in group] != list(ALL_SEEDS):
        raise RuntimeError("final candidate does not have seeds 1, 2, and 3")
    output = {
        "case": case, "seed_count": 3, "evaluation_seeds": "1,2,3",
        "evaluation_rng_mode": "continuous",
        "attention_score_mode": "fp16_fast_with_prescaled_qk_fallback",
    }
    for key in group[0]:
        if key in {"case", "family", "evaluation_seed", "status", "error"}:
            continue
        values = [finite(row.get(key)) for row in group]
        if all(value is not None for value in values):
            output[f"{key}_mean"] = statistics.mean(values)
            output[f"{key}_std"] = statistics.stdev(values)
        elif all(row.get(key) == group[0].get(key) for row in group):
            output[key] = group[0].get(key)
    return output


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    for name in (
        "compact_checkpoint", "projector_checkpoint", "cache_dir", "output_dir",
    ):
        setattr(args, name, absolute(getattr(args, name)))
    if not 0 <= args.max_mae_increase_ratio < 1:
        raise ValueError("--max-mae-increase-ratio must be in [0, 1)")
    if not 0 <= args.final_quality_slack_ratio < 1:
        raise ValueError("--final-quality-slack-ratio must be in [0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        validate(args)
        prepare_manifest(args)
    rows_path = args.output_dir / "all_runs.csv"
    rows = load_rows(rows_path) if args.resume else []

    baseline_dir = result_directory(args, "pure_nbs_compact", SEARCH_SEED)
    baseline = run_candidate(
        args, rows, "pure_nbs_compact", "baseline", {}, SEARCH_SEED,
        common_command(args, args.compact_checkpoint, baseline_dir),
    )
    baseline_mae = finite(baseline.get("mae")) if not args.dry_run else 1.0
    if not args.dry_run and baseline["status"] != "complete":
        raise RuntimeError("pure NBS baseline failed")

    for case, config in PATCH_CASES:
        directory = result_directory(args, case, SEARCH_SEED)
        run_candidate(
            args, rows, case, "patch", config, SEARCH_SEED,
            selector_command(args, args.compact_checkpoint, directory, config),
        )
    for recent_k in TOKEN_K_VALUES:
        case = f"token_k{recent_k}"
        directory = result_directory(args, case, SEARCH_SEED)
        run_candidate(
            args, rows, case, "token", {"recent_k": recent_k}, SEARCH_SEED,
            token_command(args, args.compact_checkpoint, directory, recent_k),
        )
    for gamma, threshold in SPECULATIVE_CONFIGS:
        case = f"spec_g{gamma}_t{str(threshold).replace('.', 'p')}"
        directory = result_directory(args, case, SEARCH_SEED)
        run_candidate(
            args, rows, case, "speculative",
            {"gamma": gamma, "acceptance_threshold": threshold}, SEARCH_SEED,
            speculative_command(
                args, args.compact_checkpoint, directory, gamma, threshold,
            ),
        )
    if args.dry_run:
        print(
            f"Dry run: 1 baseline + {len(PATCH_CASES)} patch + "
            f"{len(TOKEN_K_VALUES)} token + {len(SPECULATIVE_CONFIGS)} spec",
            flush=True,
        )
        return

    selected = {
        "patch": representatives(
            rows, "patch", baseline_mae, args.max_mae_increase_ratio
        ),
        "token": representatives(
            rows, "token", baseline_mae, args.max_mae_increase_ratio
        ),
        "speculative": representatives(
            rows, "speculative", baseline_mae, args.max_mae_increase_ratio
        ),
    }
    write_json(args.output_dir / "selected_independent_candidates.json", selected)

    for patch, token, speculative in full_stack_combinations(selected):
        case = stack_case(patch, token, speculative)
        directory = result_directory(args, case, SEARCH_SEED)
        config = {
            "patch_case": patch["case"],
            "recent_k": int(token["recent_k"]),
            "gamma": int(speculative["gamma"]),
            "acceptance_threshold": float(
                speculative["acceptance_threshold"]
            ),
        }
        run_candidate(
            args, rows, case, "full_stack", config, SEARCH_SEED,
            full_stack_command(
                args, args.compact_checkpoint, directory,
                patch_config(patch), config["recent_k"], config["gamma"],
                config["acceptance_threshold"],
            ),
        )

    final = choose_final(rows, args.final_quality_slack_ratio)
    write_json(args.output_dir / "selected_final_candidate.json", final)
    config = {
        "patch_case": final["patch_case"],
        "recent_k": int(final["recent_k"]),
        "gamma": int(final["gamma"]),
        "acceptance_threshold": float(final["acceptance_threshold"]),
    }
    for seed in CONFIRMATION_SEEDS:
        directory = result_directory(args, final["case"], seed)
        run_candidate(
            args, rows, final["case"], "full_stack", config, seed,
            full_stack_command(
                args, args.compact_checkpoint, directory,
                dict(PATCH_CASES)[final["patch_case"]], config["recent_k"],
                config["gamma"], config["acceptance_threshold"],
            ),
        )
    summary = summarize_final(rows, final["case"])
    write_rows(args.output_dir / "final_three_seed_summary.csv", [summary])
    print(f"All runs: {rows_path}", flush=True)
    print(
        f"Final three-seed summary: "
        f"{args.output_dir / 'final_three_seed_summary.csv'}", flush=True,
    )


if __name__ == "__main__":
    main()
