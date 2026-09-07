"""Run a tiny, read-only FP16 range audit on the ABR NBS seed-1 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ABR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ABR_ROOT.parent
DEFAULT_CHECKPOINT = ABR_ROOT / (
    "data/ft_plms/llama_base/"
    "adaptive_bitrate_streaming_artifacts_exp_pools_ss_None/"
    "rank_32_nbs_v19_budget1536_w_20_gamma_1.0_sfd_256_lr_0.0002_"
    "wd_0.0001_warm_500_epochs_80_seed_1/early_stop_-1_best_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=REPO_ROOT / "downloaded_plms/llama/base",
    )
    parser.add_argument(
        "--exp-pool-path",
        type=Path,
        default=ABR_ROOT / "artifacts/exp_pools/exp_pool.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ABR_ROOT / "artifacts/results/nbs_c1536_seed1_range_audit.json",
    )
    parser.add_argument("--threshold", type=float, default=60000.0)
    parser.add_argument("--trace-num", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "run_plm.py",
        "--test",
        "--nbs-v19",
        "--nbs-rank-budget", "1536",
        "--nbs-rank-config", "configs/nbs_v19_rank_config.json",
        "--fp16",
        "--seed", "1",
        "--lora-seed", "1",
        "--data-seed", "1",
        "--plm-type", "llama",
        "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--model-dir", str(args.checkpoint.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--rank", "32",
        "--lr", "0.0002",
        "--lr-schedule", "cosine",
        "--warmup-steps", "500",
        "--num-epochs", "80",
        "--trace", "fcc-test",
        "--trace-num", str(args.trace_num),
        "--video", "video1",
        "--fixed-order",
        "--device", args.device,
        "--device-out", args.device,
        "--temporal-selector", "none",
        "--token-selector", "none",
        "--speculative-draft-steps", "0",
    ]


def main() -> None:
    args = parse_args()
    cmd = command(args)
    print("Range audit mode: detect-only (checkpoint weights are not modified)")
    print("Command:", " ".join(cmd))
    if args.dry_run:
        return
    for path, label in (
        (args.checkpoint, "checkpoint"),
        (args.base_model_dir / "config.json", "base model"),
        (args.exp_pool_path, "experience pool"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    env = os.environ.copy()
    env["ABR_LORA_RANGE_AUDIT_PATH"] = str(args.output.resolve())
    env["ABR_LORA_RANGE_AUDIT_THRESHOLD"] = str(args.threshold)
    subprocess.run(cmd, cwd=ABR_ROOT, env=env, check=True)
    summary = json.loads(args.output.read_text(encoding="utf-8"))["summary"]
    print(f"Range audit saved at: {args.output.resolve()}")
    print(
        "Audit summary: "
        f"would_clamp={summary['would_clamp_calls']}, "
        f"input_nonfinite={summary['input_nonfinite_calls']}, "
        f"precast_nonfinite={summary['precast_nonfinite_calls']}, "
        f"max_precast_absmax={summary['max_precast_absmax']:.6g}"
    )


if __name__ == "__main__":
    main()
