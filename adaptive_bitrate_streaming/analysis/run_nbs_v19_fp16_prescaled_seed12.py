"""Evaluate the five ABR settings for seeds 1 and 2 with safe FP16 QK.

This is intentionally separate from the seed-3 pipeline so existing results
are never overwritten.  It performs inference only and never modifies the
NBS checkpoint.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics


SEED3_PATH = Path(__file__).with_name("run_nbs_v19_fp16_prescaled_seed3.py")
SPEC = importlib.util.spec_from_file_location("abr_fp16_prescaled_seed3", SEED3_PATH)
seed3_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seed3_pipeline)


sweep = seed3_pipeline.sweep
latest = seed3_pipeline.latest
SEEDS = (1, 2)
DEFAULT_OUTPUT = (
    sweep.best5.RESULTS_ROOT
    / "nbs_v19_budget1536_fp16_prescaled_qk_seed1_2"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--exp-pool-path", type=Path, required=True)
    parser.add_argument("--rank-budget", type=int, default=1536)
    parser.add_argument("--physical-rank", type=int, default=32)
    parser.add_argument(
        "--rank-config", type=Path,
        default=Path("configs/nbs_v19_rank_config.json"),
    )
    parser.add_argument("--trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nbs-compaction-rtol", type=float, default=5e-3)
    parser.add_argument("--nbs-compaction-atol", type=float, default=5e-3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


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
        "data_seeds": list(SEEDS),
        "evaluation_rng_mode": "per-episode",
        "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
        "fp16_numeric_safeguards": True,
        "fp16_selective_clamp_threshold": 60000.0,
        "nbs_compaction_rtol": args.nbs_compaction_rtol,
        "nbs_compaction_atol": args.nbs_compaction_atol,
        "specs": [
            sweep.configured_fields(item) | {"name": item["name"]}
            for item in latest.TARGET_SPECS
        ],
    }


def validate_or_write_manifest(args):
    path = args.output_dir / "pipeline_manifest.json"
    expected = {"signature": run_signature(args)}
    if path.is_file() and args.resume:
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("resume manifest differs from this configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    path.write_text(json.dumps(expected, indent=2), encoding="utf-8")


def summarize(rows):
    summaries = []
    for spec in latest.TARGET_SPECS:
        group = sorted(
            (row for row in rows if row["experiment"] == spec["name"]),
            key=lambda row: int(row["data_seed"]),
        )
        if [int(row["data_seed"]) for row in group] != list(SEEDS):
            raise ValueError(f"{spec['name']} does not have seeds 1 and 2")
        summary = {
            "experiment": spec["name"],
            "num_seeds": len(SEEDS),
            "data_seeds": "1,2",
            "evaluation_rng_mode": "per-episode",
            "attention_score_mode": "fp16_prescaled_qk_with_fp32_retry",
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


def main(argv=None):
    args = parse_args(argv)
    args.lora_seed = 1
    args.evaluation_rng_mode = "per-episode"
    args.run_tag = "per_episode_reseed_fp16_prescaled_qk"
    args.fp16_numeric_safeguards = True
    args.fp16_selective_clamp = True
    args.fp16_selective_clamp_threshold = 60000.0
    args.fp16_attention_fp32_scores = False
    args.fp16_attention_prescaled_qk = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / "fp16_prescaled_qk_seed1_2_results.csv"

    if not args.dry_run:
        sweep.best5.validate_checkpoint(args.checkpoint_dir, args.rank_budget)
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
        validate_or_write_manifest(args)

    rows = sweep.load_rows(runs_path) if args.resume else []
    for data_seed in SEEDS:
        rows = sweep.run_specs(
            args, latest.TARGET_SPECS, data_seed, runs_path, rows
        )
    if args.dry_run:
        return

    summary_path = args.output_dir / "fp16_prescaled_qk_seed1_2_summary.csv"
    sweep.write_rows(summary_path, summarize(rows))
    print(f"Seed-1/2 results: {runs_path.resolve()}")
    print(f"Two-seed summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
