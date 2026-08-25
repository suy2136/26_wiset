"""Train and test equal-capacity ABR NBS v19 (budget 8192, seed 1).

The 64 q/v LoRA modules have mean active rank 128.  Physical rank 256 leaves
room for NBS reallocation while retaining v19's legacy spectral-shadow policy.
The test stage compares NBS-only against the official rank-128 NetLLM LoRA.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ABR_ROOT / "data" / "ft_plms"
RESULTS_ROOT = ABR_ROOT / "artifacts" / "results"
DEFAULT_BASE_MODEL = ABR_ROOT.parent / "downloaded_plms" / "llama" / "base"
DEFAULT_EXP_POOL = ABR_ROOT / "artifacts" / "exp_pools" / "exp_pool.pkl"
DEFAULT_OFFICIAL_LORA = Path(
    "/workspace/abr_checkpoint_download/extracted/try_llama2_7b"
)
DEFAULT_OUTPUT = RESULTS_ROOT / "nbs_v19_budget8192_equal_capacity.csv"
RANK_BUDGET = 8192
PHYSICAL_RANK = 256
RANK_CONFIG = "configs/nbs_v19_rank_config_max256.json"


def common_args(args):
    return [
        "--fp16", "--seed", "1", "--plm-type", "llama", "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--video", args.video, "--fixed-order", "--device", args.device,
        "--device-out", args.device, "--token-selector", "none",
        "--speculative-draft-steps", "0",
    ]


def training_command(args):
    return [
        sys.executable, "run_plm.py", "--adapt", "--nbs-v19",
        *common_args(args), "--rank", str(PHYSICAL_RANK),
        "--nbs-rank-budget", str(RANK_BUDGET),
        "--nbs-rank-config", RANK_CONFIG,
        "--trace", args.train_trace, "--trace-num", str(args.trace_num),
        "--grad-accum-steps", str(args.grad_accum_steps), "--lr", str(args.lr),
        "--warmup-steps", str(args.warmup_steps),
        "--num-epochs", str(args.num_epochs),
        "--eval-per-epoch", str(args.eval_per_epoch),
    ]


def nbs_test_command(args, checkpoint):
    return [
        sys.executable, "run_plm.py", "--test", "--nbs-v19",
        *common_args(args), "--rank", str(PHYSICAL_RANK),
        "--nbs-rank-budget", str(RANK_BUDGET),
        "--nbs-rank-config", RANK_CONFIG,
        "--model-dir", str(checkpoint.resolve()), "--trace", args.test_trace,
        "--trace-num", str(args.trace_num),
    ]


def official_test_command(args):
    return [
        sys.executable, "run_plm.py", "--test", *common_args(args),
        "--rank", "128", "--model-dir", str(args.official_lora_dir.resolve()),
        "--trace", args.test_trace, "--trace-num", str(args.trace_num),
    ]


def discover_checkpoint(started_at):
    candidates = []
    for path in MODEL_ROOT.rglob("checkpoint_metadata.json"):
        if path.stat().st_mtime < started_at:
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            metadata.get("variant") == "nbs_v19"
            and metadata.get("seed") == 1
            and metadata.get("effective_rank_budget") == RANK_BUDGET
            and metadata.get("role") in ("best", "final")
        ):
            candidates.append((path, metadata))
    preferred = [item for item in candidates if item[1].get("role") == "best"]
    selected = preferred or candidates
    if not selected:
        raise RuntimeError("no new budget-8192 NBS best/final checkpoint found")
    return max(selected, key=lambda item: item[0].stat().st_mtime)[0].parent


def validate_checkpoint(checkpoint):
    required = (
        "adapter_config.json", "modules_except_plm.bin",
        "nash_rank_allocator.pt", "checkpoint_metadata.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not any((checkpoint / name).is_file() for name in (
        "adapter_model.bin", "adapter_model.safetensors"
    )):
        missing.append("adapter_model.bin or adapter_model.safetensors")
    if missing:
        raise FileNotFoundError(f"incomplete NBS checkpoint: {', '.join(missing)}")
    metadata = json.loads(
        (checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("effective_rank_budget") != RANK_BUDGET:
        raise ValueError("checkpoint rank budget is not 8192")


def newest_metrics(started_at):
    candidates = [
        path for path in RESULTS_ROOT.rglob("selector_metrics.json")
        if path.stat().st_mtime >= started_at
    ]
    if not candidates:
        raise RuntimeError("evaluation produced no selector_metrics.json")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalar_metrics(metrics):
    return {key: value for key, value in metrics.items()
            if value is None or isinstance(value, (str, int, float, bool))}


def write_results(rows, output):
    nbs = next((row for row in rows if row["experiment"] == "nbs_only"), None)
    if nbs:
        for row in rows:
            if isinstance(row.get("mean_reward"), (int, float)):
                row["mean_reward_delta_vs_nbs"] = (
                    row["mean_reward"] - nbs["mean_reward"]
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "experiment", "mean_reward", "mean_reward_delta_vs_nbs", "qoe_raw_mean",
        "mean_bitrate_mbps", "total_rebuffer_s", "mean_smoothness_mbps",
        "inference_latency_mean_ms", "inference_latency_p50_ms",
        "inference_latency_p95_ms", "rank_budget", "mean_active_rank",
        "physical_rank", "checkpoint_dir", "metrics_path", "seed",
    ]
    fields = [key for key in preferred if key in fields] + [
        key for key in fields if key not in preferred
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_test(name, command, checkpoint, args, rows):
    print(f"[{name}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    started_at = time.time() - 1.0
    subprocess.run(command, cwd=ABR_ROOT, check=True)
    metrics_path = newest_metrics(started_at)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows.append({
        "experiment": name,
        "seed": 1,
        "rank_budget": RANK_BUDGET,
        "mean_active_rank": 128,
        "physical_rank": PHYSICAL_RANK if name == "nbs_only" else 128,
        "checkpoint_dir": str(checkpoint.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        **scalar_metrics(metrics),
    })
    write_results(rows, args.output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path, default=DEFAULT_EXP_POOL)
    parser.add_argument("--official-lora-dir", type=Path, default=DEFAULT_OFFICIAL_LORA)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="skip training and test this budget-8192 checkpoint")
    parser.add_argument("--train-trace", default="fcc-valid")
    parser.add_argument("--test-trace", default="fcc-test")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--eval-per-epoch", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run:
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
        if not args.skip_official and not args.official_lora_dir.is_dir():
            raise FileNotFoundError(f"official LoRA not found: {args.official_lora_dir}")

    checkpoint = args.checkpoint_dir
    if checkpoint is None:
        command = training_command(args)
        print(f"[training] {shlex.join(command)}", flush=True)
        if args.dry_run:
            checkpoint = MODEL_ROOT / "NEW_BUDGET8192_BEST_MODEL"
        else:
            started_at = time.time() - 1.0
            subprocess.run(command, cwd=ABR_ROOT, check=True)
            checkpoint = discover_checkpoint(started_at)
            print(f"[training] selected {checkpoint}", flush=True)
    if not args.dry_run:
        validate_checkpoint(checkpoint)

    rows = []
    run_test("nbs_only", nbs_test_command(args, checkpoint), checkpoint, args, rows)
    if not args.skip_official:
        run_test("netllm_official_lora", official_test_command(args),
                 args.official_lora_dir, args, rows)
    if not args.dry_run:
        print(f"Results saved at: {args.output.resolve()}")


if __name__ == "__main__":
    main()
