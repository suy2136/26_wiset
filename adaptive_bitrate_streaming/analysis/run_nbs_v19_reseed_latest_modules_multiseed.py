"""Complete the latest ABR module ablation under episode reseeding.

The already evaluated per-episode-reseed pure-NBS rows are reused.  Temporal
K=1, token offsets 0/2/3/4/7, speculative K=5, and their latest full-stack
combination are evaluated afresh for seeds 1, 2, and 3.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics


ABR_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = Path(__file__).with_name("run_nbs_v19_module_sweep_pipeline.py")
SPEC = importlib.util.spec_from_file_location("abr_module_sweep", SWEEP_SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)

DEFAULT_RESEED_SOURCE = (
    sweep.best5.RESULTS_ROOT
    / "nbs_v19_budget1536_reseed_pure_full_multiseed"
    / "reseed_pure_full_per_seed.csv"
)
DEFAULT_OUTPUT_DIR = (
    sweep.best5.RESULTS_ROOT
    / "nbs_v19_budget1536_reseed_latest_modules_multiseed"
)
BASELINE = sweep.baseline_spec()
TEMPORAL = {
    "name": "temporal_k1", "module": "temporal",
    "temporal": True, "event_max_events": 1,
}
TOKEN = {
    "name": "token_o0_2_3_4_7", "module": "token",
    "token": True, "token_offsets": (0, 2, 3, 4, 7),
}
SPECULATIVE = {
    "name": "spec_repeat_last_k5", "module": "speculative",
    "speculative": True, "speculative_steps": 5,
    "speculative_drafter": "repeat-last",
}
FULL = {
    "name": "combined__temporal_k1__token_o0_2_3_4_7__spec_repeat_last_k5",
    "module": "combined", "temporal": True, "event_max_events": 1,
    "token": True, "token_offsets": (0, 2, 3, 4, 7),
    "speculative": True, "speculative_steps": 5,
    "speculative_drafter": "repeat-last",
}
TARGET_SPECS = (BASELINE, TEMPORAL, TOKEN, SPECULATIVE, FULL)
RUN_SPECS = TARGET_SPECS[1:]
SEEDS = (1, 2, 3)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def reusable_baselines(args):
    if not args.reseed_pure_runs.is_file():
        raise FileNotFoundError(args.reseed_pure_runs)
    rows = []
    source = read_csv(args.reseed_pure_runs)
    for seed in SEEDS:
        matches = [
            row for row in source
            if row.get("experiment") == BASELINE["name"]
            and int(float(row.get("data_seed", -1))) == seed
            and row.get("evaluation_rng_mode") == "per-episode"
        ]
        if len(matches) != 1:
            raise ValueError(f"missing unique reseeded pure-NBS seed {seed}")
        row = dict(matches[0])
        if Path(row["checkpoint_dir"]).resolve() != args.checkpoint_dir.resolve():
            raise ValueError("reused pure-NBS checkpoint differs from requested checkpoint")
        if int(float(row["rank_budget"])) != args.rank_budget:
            raise ValueError("reused pure-NBS rank budget differs")
        row["module"] = "baseline"
        rows.append(row)
    return rows


def summarize(rows):
    summaries = []
    for spec in TARGET_SPECS:
        group = sorted(
            (row for row in rows if row["experiment"] == spec["name"]),
            key=lambda row: int(row["data_seed"]),
        )
        if [int(row["data_seed"]) for row in group] != list(SEEDS):
            raise ValueError(f"{spec['name']} does not have seeds 1, 2, 3")
        summary = {
            "experiment": spec["name"], "num_seeds": 3,
            "data_seeds": "1,2,3", "evaluation_rng_mode": "per-episode",
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
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--exp-pool-path", type=Path, required=True)
    parser.add_argument("--reseed-pure-runs", type=Path, default=DEFAULT_RESEED_SOURCE)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--recompute-all", action="store_true",
        help=(
            "evaluate Pure NBS and all four module settings afresh instead "
            "of reusing the earlier pure-NBS rows"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def manifest_signature(args):
    return {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "base_model_dir": str(args.base_model_dir.resolve()),
        "exp_pool_path": str(args.exp_pool_path.resolve()),
        "reseed_pure_runs": str(args.reseed_pure_runs.resolve()),
        "evaluation_rng_mode": "per-episode",
        "run_tag": args.run_tag,
        "recompute_all": bool(args.recompute_all),
        "fp16_numeric_safeguards": True,
        "fp16_selective_clamp_threshold": 60000.0,
        "fp16_attention_fp32_scores": True,
        "seeds": list(SEEDS),
        "specs": [sweep.configured_fields(spec) | {"name": spec["name"]}
                  for spec in TARGET_SPECS],
    }


def validate_manifest(args):
    path = args.output_dir / "pipeline_manifest.json"
    signature = manifest_signature(args)
    if path.is_file() and args.resume:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("signature") != signature:
            raise ValueError("resume manifest differs from this configuration")
    elif path.exists() and not args.resume:
        raise FileExistsError(f"output exists: {path}; use --resume")
    path.write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    args.evaluation_rng_mode = "per-episode"
    # Compact inference replaces PEFT SVDLinear modules.  Keep the same
    # finite-range guard active across both dense and compact NBS paths.
    args.fp16_numeric_safeguards = True
    args.fp16_selective_clamp = True
    args.fp16_selective_clamp_threshold = 60000.0
    args.fp16_attention_fp32_scores = True
    args.run_tag = (
        "per_episode_reseed_fp32_attention"
        if args.recompute_all else "per_episode_reseed"
    )
    args.confirmation_seeds = list(SEEDS)
    args.combined_finalists = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / "reseed_latest_modules_per_seed.csv"

    if args.dry_run:
        rows = []
    else:
        sweep.best5.validate_checkpoint(args.checkpoint_dir, args.rank_budget)
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
        validate_manifest(args)
        rows = [] if args.recompute_all else reusable_baselines(args)
        if args.resume:
            prior = sweep.load_rows(runs_path)
            if args.recompute_all:
                rows.extend(prior)
            else:
                rows.extend(
                    row for row in prior
                    if row.get("experiment") != BASELINE["name"]
                )
        sweep.write_rows(runs_path, rows)

    selected_specs = TARGET_SPECS if args.recompute_all else RUN_SPECS
    for seed in SEEDS:
        rows = sweep.run_specs(args, selected_specs, seed, runs_path, rows)
    if args.dry_run:
        return
    summary = summarize(rows)
    summary_path = args.output_dir / "reseed_latest_modules_summary.csv"
    sweep.write_rows(summary_path, summary)
    print(f"Per-seed results: {runs_path.resolve()}")
    print(f"Three-seed summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
