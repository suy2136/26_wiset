"""Train and evaluate VP NBS rank-bound ablations at total rank budget 512.

All artifacts are additive.  Existing checkpoints and experiment results are
only read; each ablation writes to a newly timestamped training run and to its
own directory below ``--output-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import run_server1_vp_data3_pipeline as vp
from analysis.resolve_checkpoint_alias import resolve_checkpoint


TARGET_BUDGET = 512
TRAINING_DATA_SEED = 1
MODULE_COUNT = 64
VARIANT = "nbs_v19_data1"
VP_RUN_ROOT = REPO_ROOT / "viewport_prediction/data/experiment_runs/netllm_vs_nbs"
DEFAULT_OUTPUT = VP_RUN_ROOT / "vp_nbs_rank_bound_ablation_c512_data1"
SPECS = (
    ("min4_max32", 4, 32, "configs/adalora_rank_config_llama7b_min4_max32.json"),
    ("min6_max32", 6, 32, "configs/adalora_rank_config_llama7b_min6_max32.json"),
    ("min2_max16", 2, 16, "configs/adalora_rank_config_llama7b_min2_max16.json"),
    ("min2_max12", 2, 12, "configs/adalora_rank_config_llama7b_min2_max12.json"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latency-warmup-steps", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def signature() -> dict:
    return {
        "pipeline": "vp_nbs_rank_bound_ablation_v2",
        "target_budget": TARGET_BUDGET,
        "training_seed": 1,
        "lora_seed": 1,
        "training_data_seed": TRAINING_DATA_SEED,
        "evaluation_seeds_and_data_seeds": [1, 2, 3],
        "evaluation_rng_mode": "continuous",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "inference": "compact_pure_nbs",
        "specs": [
            {"name": name, "min_rank": minimum, "max_rank": maximum,
             "physical_rank": maximum, "rank_config": config}
            for name, minimum, maximum, config in SPECS
        ],
    }


def load_state(args):
    path = args.output_dir / "pipeline_state.json"
    expected = signature()
    if path.is_file():
        if not args.resume:
            raise FileExistsError(f"output exists: {args.output_dir}; use --resume")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("signature") != expected:
            previous = state.get("signature", {})
            # v1 incorrectly treated physical rank as globally fixed at 32.
            # Preserve completed max32 experiments while migrating the same
            # four-spec queue to per-spec physical widths for max16/max12.
            migratable = (
                previous.get("pipeline") == "vp_nbs_rank_bound_ablation_v1"
                and previous.get("target_budget") == TARGET_BUDGET
                and previous.get("training_data_seed") == TRAINING_DATA_SEED
                and [
                    (item.get("name"), item.get("min_rank"), item.get("max_rank"),
                     item.get("rank_config"))
                    for item in previous.get("specs", [])
                ] == [
                    (name, minimum, maximum, config)
                    for name, minimum, maximum, config in SPECS
                ]
            )
            if not migratable:
                raise ValueError("resume state differs from the current configuration")
            state["signature"] = expected
            atomic_json(path, state)
            print("Migrated rank-bound pipeline state from v1 to v2", flush=True)
        return path, state
    state = {"signature": expected, "experiments": {}}
    if not args.dry_run:
        atomic_json(path, state)
    return path, state


def validate_config(config: Path, minimum: int, maximum: int) -> None:
    values = json.loads(config.read_text(encoding="utf-8"))
    if len(values) != 2:
        raise ValueError(f"expected q/v rank rules in {config}, got {len(values)}")
    for pattern, bounds in values.items():
        actual = (int(bounds["min_rank"]), int(bounds["max_rank"]))
        if actual != (minimum, maximum):
            raise ValueError(
                f"rank bounds mismatch in {config}: {pattern}={actual}, "
                f"expected {(minimum, maximum)}"
            )
    if not minimum * MODULE_COUNT <= TARGET_BUDGET <= maximum * MODULE_COUNT:
        raise ValueError(
            f"budget {TARGET_BUDGET} is infeasible for {MODULE_COUNT} modules "
            f"with bounds [{minimum}, {maximum}]"
        )


def parse_env(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def allocator_ranks(checkpoint: Path) -> list[int]:
    import torch
    state = torch.load(checkpoint / "nash_rank_allocator.pt", map_location="cpu")
    ranks = state.get("ranks") or state.get("current_ranks") or {}
    if not isinstance(ranks, dict) or len(ranks) != MODULE_COUNT:
        raise ValueError(
            f"expected {MODULE_COUNT} allocator ranks, got "
            f"{len(ranks) if isinstance(ranks, dict) else type(ranks).__name__}"
        )
    return [vp.vp_lora.active_rank(value) for value in ranks.values()]


def validate_checkpoint(checkpoint: Path, minimum: int, maximum: int) -> list[int]:
    resolved = resolve_checkpoint(checkpoint)
    if not vp.checkpoint_complete(resolved):
        raise FileNotFoundError(f"incomplete checkpoint: {resolved}")
    vp.inspect_budget("nbs", resolved, require_exact=True)
    adapter = json.loads((resolved / "adapter_config.json").read_text(encoding="utf-8"))
    physical_rank = int(adapter.get("init_r", adapter.get("r", -1)))
    if physical_rank != maximum:
        raise ValueError(
            f"checkpoint physical rank is {physical_rank}, expected max_rank {maximum}"
        )
    ranks = allocator_ranks(resolved)
    if sum(ranks) != TARGET_BUDGET:
        raise ValueError(f"actual allocator rank total is {sum(ranks)}, expected 512")
    outside = [rank for rank in ranks if not minimum <= rank <= maximum]
    if outside:
        raise ValueError(
            f"allocator ranks violate [{minimum}, {maximum}]: {sorted(set(outside))}"
        )
    return ranks


def checkpoint_from_metadata(metadata_path: Path, config: Path,
                             minimum: int, maximum: int) -> Path | None:
    metadata = parse_env(metadata_path)
    if metadata.get("variant") != VARIANT:
        return None
    if int(metadata.get("data_seed", -1)) != TRAINING_DATA_SEED:
        return None
    if int(metadata.get("rank_budget", -1)) != TARGET_BUDGET:
        return None
    if int(metadata.get("rank", -1)) != maximum:
        return None
    recorded = Path(metadata.get("rank_config", ""))
    if not recorded.is_absolute():
        recorded = REPO_ROOT / recorded
    if recorded.resolve() != config.resolve():
        return None
    candidate = Path(metadata.get("final_nbs_model", ""))
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        validate_checkpoint(candidate, minimum, maximum)
    except (FileNotFoundError, ValueError):
        return None
    return resolve_checkpoint(candidate).resolve()


def recover_checkpoint(config: Path, minimum: int, maximum: int) -> Path | None:
    history = VP_RUN_ROOT / VARIANT
    metadata_paths = sorted(
        history.glob("*/metadata.env"), key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if history.is_dir() else []
    for metadata_path in metadata_paths:
        checkpoint = checkpoint_from_metadata(
            metadata_path, config, minimum, maximum,
        )
        if checkpoint is not None:
            return checkpoint
    return None


def train(args, config: Path, minimum: int, maximum: int) -> Path:
    if args.resume:
        recovered = recover_checkpoint(config, minimum, maximum)
        if recovered is not None:
            print(f"Recovered matching checkpoint: {recovered}", flush=True)
            return recovered
    command = ["bash", "scripts/run_netllm_experiment.sh", VARIANT]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return Path("/dry-run/checkpoint")
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{environment.get('PATH', '')}",
        "SKIP_EVALUATION": "1",
        "SKIP_VISUALIZATION": "1",
        "SAVE_PERIODIC_CHECKPOINTS": "0",
        "VP_FP16_PRESCALED_QK": "1",
        "VP_TOTAL_RANK_BUDGET": str(TARGET_BUDGET),
        "VP_NBS_RANK_CONFIG": str(config.relative_to(REPO_ROOT)),
        # In VP NBS, configured max_rank is the physical AdaLoRA width.
        "VP_NBS_TARGET_RANK": str(maximum),
    })
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    checkpoint = recover_checkpoint(config, minimum, maximum)
    if checkpoint is None:
        raise FileNotFoundError(
            f"training completed but no exact checkpoint matched config {config}"
        )
    return checkpoint


def experiment_args(args, output_dir: Path):
    return argparse.Namespace(
        output_dir=output_dir,
        device=args.device,
        latency_warmup_steps=args.latency_warmup_steps,
        resume=args.resume,
        dry_run=args.dry_run,
        tuned_projector=Path("/unused/projector.pth"),
        patch_cache=Path("/unused/patch-cache"),
    )


def run_experiment(args, state, state_path: Path, spec) -> None:
    name, minimum, maximum, config_name = spec
    config = (REPO_ROOT / config_name).resolve()
    validate_config(config, minimum, maximum)
    record = state["experiments"].setdefault(name, {})
    if record.get("status") == "complete":
        print(f"[{name}] already complete; skipping", flush=True)
        return
    record.update({"status": "running", "started_at": time.time()})
    if not args.dry_run:
        atomic_json(state_path, state)
    try:
        saved = record.get("checkpoint")
        checkpoint = resolve_checkpoint(Path(saved)) if saved else None
        if checkpoint is None or not vp.checkpoint_complete(checkpoint):
            checkpoint = train(args, config, minimum, maximum)
            record["checkpoint"] = str(checkpoint.resolve())
            if not args.dry_run:
                atomic_json(state_path, state)
        if args.dry_run:
            print(
                f"Would validate bounds [{minimum}, {maximum}], compact to "
                f"rank {TARGET_BUDGET} (physical rank {maximum}), and evaluate seeds 1,2,3 with "
                "FP16 prescaled Q/K.",
                flush=True,
            )
            return
        ranks = [] if args.dry_run else validate_checkpoint(
            checkpoint, minimum, maximum,
        )
        local_args = experiment_args(args, args.output_dir / name)
        compact = vp.compact_nbs(local_args, checkpoint)
        vp.evaluate_modules(local_args, compact, vp.MODULE_CASES[:1])
        record.update({
            "status": "complete", "finished_at": time.time(),
            "checkpoint": str(checkpoint.resolve()),
            "compact_checkpoint": str(compact.resolve()),
            "actual_rank_total": sum(ranks) if ranks else TARGET_BUDGET,
            "actual_rank_min": min(ranks) if ranks else minimum,
            "actual_rank_max": max(ranks) if ranks else maximum,
            "physical_rank": maximum,
        })
    except Exception as error:
        record.update({
            "status": "failed", "finished_at": time.time(),
            "error_type": type(error).__name__, "error": str(error),
        })
        print(f"[{name}] FAILED: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
    if not args.dry_run:
        atomic_json(state_path, state)


def write_summary(args, state) -> None:
    rows = []
    for name, minimum, maximum, config in SPECS:
        record = state["experiments"].get(name, {})
        summary = load_rows(args.output_dir / name / "nbs_modules/three_seed_summary.csv")
        if record.get("status") != "complete" or not summary:
            continue
        row = dict(summary[0])
        row.update({
            "experiment": name, "configured_min_rank": minimum,
            "configured_max_rank": maximum, "rank_config": config,
            "physical_rank": maximum,
            "actual_rank_total": record["actual_rank_total"],
            "actual_rank_min": record["actual_rank_min"],
            "actual_rank_max": record["actual_rank_max"],
            "checkpoint": record["checkpoint"],
        })
        rows.append(row)
    vp.write_rows(args.output_dir / "three_seed_summary.csv", rows)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vp.TARGET_BUDGET = TARGET_BUDGET
    vp.TRAINING_DATA_SEED = TRAINING_DATA_SEED
    state_path, state = load_state(args)
    for spec in SPECS:
        print(f"\n===== {spec[0]} =====", flush=True)
        run_experiment(args, state, state_path, spec)
    if not args.dry_run:
        write_summary(args, state)
        failed = [name for name, value in state["experiments"].items()
                  if value.get("status") != "complete"]
        print(f"Pipeline state: {state_path}", flush=True)
        print(f"Failed experiments: {failed}", flush=True)
        print(f"OUTPUT={args.output_dir}", flush=True)
        raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
