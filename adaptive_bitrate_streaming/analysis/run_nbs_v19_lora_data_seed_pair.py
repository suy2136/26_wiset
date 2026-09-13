"""Evaluate the seed-3-trained ABR NBS checkpoint and pair it with seed 1.

The five legacy-FP16 seed-1 rows are read, never rerun.  The same five
module settings are evaluated once with the data-seed-3-trained checkpoint
and data seed 3.  Per-run rows and a two-checkpoint mean/sample-SD summary
are written without enabling FP32 attention-score inference.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics

ABR_ROOT = Path(__file__).resolve().parents[1]
LATEST_SCRIPT = Path(__file__).with_name(
    "run_nbs_v19_reseed_latest_modules_multiseed.py"
)
LATEST_SPEC = importlib.util.spec_from_file_location(
    "abr_reseed_latest_modules", LATEST_SCRIPT
)
latest = importlib.util.module_from_spec(LATEST_SPEC)
LATEST_SPEC.loader.exec_module(latest)
sweep = latest.sweep
DEFAULT_SEED1_RESULTS = (
    ABR_ROOT / "artifacts" / "results"
    / "nbs_v19_budget1536_reseed_latest_modules_multiseed"
    / "reseed_latest_modules_per_seed.csv"
)
DEFAULT_OUTPUT_DIR = (
    ABR_ROOT / "artifacts" / "results"
    / "nbs_v19_budget1536_lora_data_seed_1_3"
)
TARGET_SPECS = latest.TARGET_SPECS
TARGET_NAMES = tuple(spec["name"] for spec in TARGET_SPECS)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def legacy_seed1_rows(path, rank_budget):
    if not path.is_file():
        raise FileNotFoundError(f"legacy seed-1 results not found: {path}")
    source = read_csv(path)
    selected = []
    for name in TARGET_NAMES:
        matches = [
            row for row in source
            if row.get("experiment") == name
            and int(float(row.get("data_seed", -1))) == 1
        ]
        if len(matches) != 1:
            raise ValueError(f"legacy results need one seed-1 row for {name}")
        row = dict(matches[0])
        if int(float(row.get("rank_budget", -1))) != rank_budget:
            raise ValueError(f"legacy seed-1 rank budget differs for {name}")
        if "fp32_attention" in row.get("metrics_path", "").lower():
            raise ValueError(
                "seed-1 source uses FP32 attention; select the earlier FP16 result"
            )
        row["training_data_seed"] = 1
        row["evaluation_data_seed"] = 1
        row["result_role"] = "reused_legacy_fp16"
        selected.append(row)
    rng_modes = {row.get("evaluation_rng_mode", "continuous") for row in selected}
    if len(rng_modes) != 1:
        raise ValueError("legacy seed-1 rows use inconsistent RNG modes")
    return selected, rng_modes.pop()


def validate_seed3_checkpoint(path, rank_budget):
    required = (
        "adapter_config.json", "modules_except_plm.bin",
        "nash_rank_allocator.pt", "checkpoint_metadata.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if not any((path / name).is_file() for name in (
        "adapter_model.bin", "adapter_model.safetensors"
    )):
        missing.append("adapter_model.bin or adapter_model.safetensors")
    if missing:
        raise FileNotFoundError(
            f"incomplete seed-3 NBS checkpoint: {', '.join(missing)}"
        )
    metadata = json.loads(
        (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    if not str(metadata.get("variant", "")).startswith("nbs_v19"):
        raise ValueError("checkpoint is not an NBS v19 variant")
    if int(metadata.get("effective_rank_budget", -1)) != rank_budget:
        raise ValueError("seed-3 checkpoint rank budget differs")
    recorded = metadata.get("data_seed", metadata.get("training_data_seed"))
    path_marks_seed3 = re.search(r"(?:data_)?seed_?3(?:\D|$)", str(path))
    if recorded not in (3, "3") and not path_marks_seed3:
        raise ValueError(
            "checkpoint is not labelled as a data-seed-3 training result"
        )
    return metadata


def summarize(rows):
    summaries = []
    for spec in TARGET_SPECS:
        group = [row for row in rows if row["experiment"] == spec["name"]]
        seeds = sorted(int(row["evaluation_data_seed"]) for row in group)
        if seeds != [1, 3] or len(group) != 2:
            raise ValueError(f"{spec['name']} needs exactly data seeds 1 and 3")
        summary = {
            "experiment": spec["name"],
            "num_trained_checkpoints": 2,
            "training_data_seeds": "1,3",
            "evaluation_data_seeds": "1,3",
            "attention_mode": "legacy_fp16_scores",
            **sweep.configured_fields(spec),
        }
        for metric in sweep.SUMMARY_METRICS:
            values = [sweep.finite_number(row.get(metric)) for row in group]
            if all(value is not None for value in values):
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values)
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
    parser.add_argument("--seed3-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed1-results", type=Path, default=DEFAULT_SEED1_RESULTS)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--exp-pool-path", type=Path, required=True)
    parser.add_argument("--rank-budget", type=int, default=1536)
    parser.add_argument("--physical-rank", type=int, default=32)
    parser.add_argument(
        "--rank-config", type=Path,
        default=Path("configs/nbs_v19_rank_config.json"),
    )
    parser.add_argument("--checkpoint-lora-seed", type=int, default=1)
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--nbs-compaction-rtol", type=float, default=0.05,
        help="compact-logit relative tolerance (historical data-seed-3 value)",
    )
    parser.add_argument(
        "--nbs-compaction-atol", type=float, default=0.01,
        help="compact-logit absolute tolerance (historical data-seed-3 value)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def signature(args, rng_mode):
    return {
        "seed1_results": str(args.seed1_results.resolve()),
        "seed3_checkpoint_dir": str(args.seed3_checkpoint_dir.resolve()),
        "training_data_seeds": [1, 3],
        "evaluation_data_seeds": [1, 3],
        "evaluation_rng_mode": rng_mode,
        "attention_mode": "legacy_fp16_scores",
        "fp16_attention_fp32_scores": False,
        "nbs_compaction_rtol": args.nbs_compaction_rtol,
        "nbs_compaction_atol": args.nbs_compaction_atol,
        "rank_budget": args.rank_budget,
        "physical_rank": args.physical_rank,
        "checkpoint_lora_seed": args.checkpoint_lora_seed,
        "specs": [sweep.configured_fields(spec) | {"name": spec["name"]}
                  for spec in TARGET_SPECS],
    }


def validate_or_write_manifest(args, rng_mode):
    path = args.output_dir / "pipeline_manifest.json"
    expected = {"signature": signature(args, rng_mode)}
    if path.is_file() and args.resume:
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("resume manifest differs from this configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    path.write_text(json.dumps(expected, indent=2), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    seed1, rng_mode = legacy_seed1_rows(args.seed1_results, args.rank_budget)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = args.seed3_checkpoint_dir
    args.data_seed = 3
    args.lora_seed = args.checkpoint_lora_seed
    args.evaluation_rng_mode = rng_mode
    args.run_tag = "lora_data_seed3_legacy_fp16"
    # Deliberately match the historical seed-1 path: do not enable the new
    # FP32 attention-score path or the later optional FP16 safeguards.
    args.fp16_numeric_safeguards = False
    args.fp16_selective_clamp = False
    args.fp16_attention_fp32_scores = False

    if not args.dry_run:
        validate_seed3_checkpoint(args.seed3_checkpoint_dir, args.rank_budget)
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
        validate_or_write_manifest(args, rng_mode)

    seed3_path = args.output_dir / "seed3_trained_checkpoint_results.csv"
    seed3_rows = sweep.load_rows(seed3_path) if args.resume else []
    seed3_rows = sweep.run_specs(
        args, TARGET_SPECS, 3, seed3_path, seed3_rows
    )
    if args.dry_run:
        return
    for row in seed3_rows:
        row["training_data_seed"] = 3
        row["evaluation_data_seed"] = 3
        row["result_role"] = "new_seed3_checkpoint_legacy_fp16"
    pair_rows = seed1 + seed3_rows
    runs_path = args.output_dir / "lora_data_seed_1_3_runs.csv"
    summary_path = args.output_dir / "lora_data_seed_1_3_summary.csv"
    sweep.write_rows(runs_path, pair_rows)
    sweep.write_rows(summary_path, summarize(pair_rows))
    print(f"Two-checkpoint runs: {runs_path.resolve()}")
    print(f"Two-checkpoint summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
