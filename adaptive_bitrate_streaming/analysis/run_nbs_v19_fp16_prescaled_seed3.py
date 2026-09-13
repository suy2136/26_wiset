"""Evaluate the latest five ABR settings for seed 3 with safe FP16 QK."""

import argparse
import importlib.util
import json
from pathlib import Path


ABR_ROOT = Path(__file__).resolve().parents[1]
LATEST_PATH = Path(__file__).with_name(
    "run_nbs_v19_reseed_latest_modules_multiseed.py"
)
SPEC = importlib.util.spec_from_file_location("abr_latest_modules", LATEST_PATH)
latest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latest)
sweep = latest.sweep

DEFAULT_OUTPUT = (
    sweep.best5.RESULTS_ROOT
    / "nbs_v19_budget1536_fp16_prescaled_qk_seed3"
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
        "data_seed": 3,
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


def main(argv=None):
    args = parse_args(argv)
    args.data_seed = 3
    args.lora_seed = 1
    args.evaluation_rng_mode = "per-episode"
    args.run_tag = "per_episode_reseed_fp16_prescaled_qk"
    args.fp16_numeric_safeguards = True
    args.fp16_selective_clamp = True
    args.fp16_selective_clamp_threshold = 60000.0
    args.fp16_attention_fp32_scores = False
    args.fp16_attention_prescaled_qk = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "fp16_prescaled_qk_seed3_results.csv"

    if not args.dry_run:
        sweep.best5.validate_checkpoint(args.checkpoint_dir, args.rank_budget)
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
        validate_or_write_manifest(args)

    rows = sweep.load_rows(output) if args.resume else []
    rows = sweep.run_specs(
        args, latest.TARGET_SPECS, 3, output, rows
    )
    if args.dry_run:
        return
    print(f"Seed-3 results: {output.resolve()}")


if __name__ == "__main__":
    main()
