"""Train the recommended ABR NBS experiments sequentially in C-A-D order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ABR_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ABR_ROOT / "data" / "ft_plms"
DEFAULT_BASE_MODEL = ABR_ROOT.parent / "downloaded_plms" / "llama" / "base"
DEFAULT_EXP_POOL = ABR_ROOT / "artifacts" / "exp_pools" / "exp_pool.pkl"
DEFAULT_STATE = (
    ABR_ROOT / "artifacts" / "results" / "nbs_v19_cad_training_state.json"
)

# Keep this tuple ordered: subprocess execution follows this exact order.
EXPERIMENTS = (
    {
        "name": "C", "rank_budget": 1536, "physical_rank": 32,
        "lr": 2e-4, "lr_schedule": "cosine",
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "A", "rank_budget": 2048, "physical_rank": 32,
        "lr": 1e-4, "lr_schedule": "constant",
        "rank_config": "configs/nbs_v19_rank_config.json",
    },
    {
        "name": "D", "rank_budget": 3072, "physical_rank": 64,
        "lr": 2e-4, "lr_schedule": "cosine",
        "rank_config": "configs/nbs_v19_rank_config_max64.json",
    },
)


def build_training_command(args, experiment):
    warmup_steps = (
        args.cosine_warmup_steps
        if experiment["lr_schedule"] == "cosine"
        else args.warmup_steps
    )
    return [
        sys.executable, "run_plm.py", "--adapt", "--nbs-v19", "--fp16",
        "--seed", "1", "--plm-type", "llama", "--plm-size", "base",
        "--plm-dir", str(args.base_model_dir.resolve()),
        "--exp-pool-path", str(args.exp_pool_path.resolve()),
        "--rank", str(experiment["physical_rank"]),
        "--nbs-rank-budget", str(experiment["rank_budget"]),
        "--nbs-rank-config", experiment["rank_config"],
        "--lr", str(experiment["lr"]),
        "--lr-schedule", experiment["lr_schedule"],
        "--warmup-steps", str(warmup_steps),
        "--num-epochs", str(args.num_epochs),
        "--eval-per-epoch", str(args.eval_per_epoch),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--early-stopping-min-epochs", str(args.early_stopping_min_epochs),
        "--early-stopping-min-delta", str(args.early_stopping_min_delta),
        "--plateau-lr-patience", str(args.plateau_lr_patience),
        "--plateau-lr-factor", str(args.plateau_lr_factor),
        "--plateau-min-lr", str(args.plateau_min_lr),
        "--trace", args.train_trace, "--trace-num", str(args.trace_num),
        "--video", args.video, "--fixed-order",
        "--device", args.device, "--device-out", args.device,
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--token-selector", "none", "--speculative-draft-steps", "0",
        "--save-checkpoint-per-epoch", "10",
        "--checkpoint-retention", "best-latest",
    ]


def discover_best_checkpoint(experiment, started_at):
    marker = (
        f"rank_{experiment['physical_rank']}_nbs_v19_"
        f"budget{experiment['rank_budget']}_"
    )
    candidates = []
    for path in MODEL_ROOT.rglob("checkpoint_metadata.json"):
        if marker not in str(path) or path.stat().st_mtime < started_at:
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("role") == "best" and (
            metadata.get("effective_rank_budget") == experiment["rank_budget"]
        ):
            candidates.append(path.parent)
    if not candidates:
        raise RuntimeError(
            f"experiment {experiment['name']} produced no new best checkpoint"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_state(path, resume, experiments=EXPERIMENTS):
    order = [item["name"] for item in experiments]
    if not resume or not path.is_file():
        return {"order": order, "runs": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("order") != order:
        raise ValueError(
            f"saved state does not use {'-'.join(order)} experiment order"
        )
    return state


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def parse_args(argv=None, default_state=DEFAULT_STATE):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--exp-pool-path", type=Path, default=DEFAULT_EXP_POOL)
    parser.add_argument("--train-trace", default="fcc-valid")
    parser.add_argument("--trace-num", type=int, default=100)
    parser.add_argument("--video", default="video1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument(
        "--cosine-warmup-steps", type=int, default=500,
        help="shorter warmup for C/D so cosine decay begins before early stopping",
    )
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--eval-per-epoch", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.003)
    parser.add_argument("--plateau-lr-patience", type=int, default=5)
    parser.add_argument("--plateau-lr-factor", type=float, default=0.5)
    parser.add_argument("--plateau-min-lr", type=float, default=1e-6)
    parser.add_argument("--state-file", type=Path, default=default_state)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run_experiments(args, experiments=EXPERIMENTS):
    if not args.dry_run:
        if not (args.base_model_dir / "config.json").is_file():
            raise FileNotFoundError(f"base model not found: {args.base_model_dir}")
        if not args.exp_pool_path.is_file():
            raise FileNotFoundError(f"experience pool not found: {args.exp_pool_path}")
    state = load_state(args.state_file, args.resume, experiments)
    for experiment in experiments:
        name = experiment["name"]
        if name in state["runs"]:
            print(f"[{name}] already complete; skipping", flush=True)
            continue
        command = build_training_command(args, experiment)
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        started_at = time.time() - 1.0
        subprocess.run(command, cwd=ABR_ROOT, check=True)
        checkpoint = discover_best_checkpoint(experiment, started_at)
        state["runs"][name] = {
            **experiment,
            "checkpoint_dir": str(checkpoint.resolve()),
            "completed_at": time.time(),
        }
        save_state(args.state_file, state)
        print(f"[{name}] best checkpoint: {checkpoint}", flush=True)
    if not args.dry_run:
        print(f"Training state saved at: {args.state_file.resolve()}")


def main(argv=None):
    run_experiments(parse_args(argv), EXPERIMENTS)


if __name__ == "__main__":
    main()
