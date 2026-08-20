"""Benchmark the reference repository's direct autoregressive NetLLM path.

This harness deliberately imports all model and pipeline code from a separate
checkout of the reference repository.  Only the timing/output harness lives in
the current project, allowing both implementations to be measured on the same
GPU without copying the Llama weights or changing the reference source tree.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum test samples including warm-up; 0 evaluates the full split.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=32)
    return parser.parse_args()


ARGS = parse_args()
REFERENCE_REPO = os.path.abspath(ARGS.reference_repo)
MODEL_PATH = os.path.abspath(ARGS.model_path)

if not os.path.isdir(REFERENCE_REPO):
    raise FileNotFoundError(f"reference repository not found: {REFERENCE_REPO}")
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"plain LoRA checkpoint not found: {MODEL_PATH}")
if ARGS.warmup_steps < 0 or ARGS.max_samples < 0:
    raise ValueError("warmup-steps and max-samples must be non-negative")

# config.py derives all relative model/dataset paths from the current working
# directory.  Change directories before importing anything from the reference.
os.chdir(REFERENCE_REPO)
sys.path.insert(0, REFERENCE_REPO)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config import cfg  # noqa: E402
from dataset.load_dataset import create_dataset  # noqa: E402
from models.low_rank import peft_model  # noqa: E402
from models.networking_head import NetworkingHead  # noqa: E402
from models.pipeline import Pipeline  # noqa: E402
from utils.normalize import normalize_data  # noqa: E402
from utils.plms_utils import load_plm  # noqa: E402


def percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def git_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REFERENCE_REPO, text=True
    ).strip()


def load_reference_pipeline():
    if ARGS.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is False")

    model_dir = os.path.join(cfg.plms_dir, "llama", "base")
    plm, _, _ = load_plm(
        "llama",
        model_dir,
        plm_size="base",
        device_input_side=ARGS.device,
        device_output_side=ARGS.device,
        device_middle_side=None,
        torch_dtype=torch.float16,
    )
    plm = peft_model(plm, "llama", ARGS.rank)

    # The pinned reference PEFT wrapper promotes all one-dimensional weights to
    # fp32.  Its own real-7B fp16 evaluation restores RMSNorm weights to fp16;
    # reproduce that published compatibility fix without changing inference.
    for name, parameter in plm.named_parameters():
        if "norm.weight" in name:
            parameter.data = parameter.data.to(torch.float16)

    head = NetworkingHead(input_dim=4096, output_dim=3, fut_window=20).to(ARGS.device)
    plm.set_networking_head(head)
    pipeline = Pipeline(
        plm,
        fut_window=20,
        device=ARGS.device,
        embed_size=4096,
        frequency=5,
        multimodal_mode="none",
        dataset="Jin2022",
        vit_model=None,
    )

    pipeline.plm.load_adapter(MODEL_PATH, adapter_name="default")
    pipeline.plm.set_adapter("default")
    state = torch.load(
        os.path.join(MODEL_PATH, "modules_except_plm.bin"),
        map_location=ARGS.device,
    )
    state = {key.replace("task_head", "networking_head"): value for key, value in state.items()}
    incompatible = pipeline.modules_except_plm.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible non-PLM checkpoint keys: {incompatible}")
    pipeline.eval()
    return pipeline


def main():
    pipeline = load_reference_pipeline()
    test_dataset = create_dataset("Jin2022", include=["test"])[0]
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    elapsed_ms = []
    records = []
    total_calls = 0
    if ARGS.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for test_step, (history, future, video_user_info) in enumerate(loader, start=1):
            if ARGS.max_samples and test_step > ARGS.max_samples:
                break
            history = normalize_data(history.to(ARGS.device), "Jin2022")
            future = future.to(ARGS.device)
            total_calls = test_step

            if history.is_cuda:
                torch.cuda.synchronize(history.device)
            started_at = time.perf_counter()
            pipeline.inference(history, future, video_user_info)
            if history.is_cuda:
                torch.cuda.synchronize(history.device)
            duration_ms = (time.perf_counter() - started_at) * 1000.0

            if test_step > ARGS.warmup_steps:
                elapsed_ms.append(duration_ms)
                records.append(
                    {
                        "test_step": test_step,
                        "video": int(video_user_info[0]),
                        "user": int(video_user_info[1]),
                        "timestep": int(video_user_info[2]),
                        "latency_ms": duration_ms,
                    }
                )
            if test_step % 200 == 0:
                print(f"measured {test_step}/{min(len(test_dataset), ARGS.max_samples or len(test_dataset))}")

    if not elapsed_ms:
        raise RuntimeError("no timed calls; max-samples must exceed warmup-steps")

    output_path = os.path.abspath(ARGS.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    detail_path = os.path.splitext(output_path)[0] + "_per_sample.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    gpu_name = torch.cuda.get_device_name(0) if ARGS.device.startswith("cuda") else None
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if ARGS.device.startswith("cuda")
        else None
    )
    summary = {
        "implementation": "reference_repository_direct_autoregressive",
        "reference_repository": REFERENCE_REPO,
        "reference_commit": git_revision(),
        "checkpoint": MODEL_PATH,
        "device": ARGS.device,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "dtype": "float16",
        "batch_size": 1,
        "history_window": 10,
        "future_window": 20,
        "rank": ARGS.rank,
        "warmup_calls": min(ARGS.warmup_steps, total_calls),
        "measured_calls": len(elapsed_ms),
        "total_calls": total_calls,
        "mean_ms": statistics.fmean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "p95_ms": percentile(elapsed_ms, 0.95),
        "std_ms": statistics.pstdev(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
        "peak_memory_mb": peak_memory_mb,
        "measurement_scope": {
            "operation": "reference Pipeline.inference",
            "includes_model_loading": False,
            "includes_data_loading": False,
            "includes_normalization": False,
            "includes_metric_computation": False,
            "cuda_synchronized": True,
        },
        "per_sample_csv": detail_path,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(
        "Reference direct AR latency: mean={:.3f} ms, median={:.3f} ms, "
        "p95={:.3f} ms over {} calls".format(
            summary["mean_ms"],
            summary["median_ms"],
            summary["p95_ms"],
            summary["measured_calls"],
        )
    )
    print("Summary saved at", output_path)
    print("Per-sample latency saved at", detail_path)


if __name__ == "__main__":
    main()
