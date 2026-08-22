"""Precompute task-specific EVA activation PCA for NetLLM.

This script only creates EVA artifacts.  It does not alter or train existing
LoRA, AdaLoRA, or NBS models; adapter construction is a later integration step.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from dataset.load_dataset import create_dataset
from models.eva_initializer import EvaActivationCollector, save_eva_state
from models.networking_head import NetworkingHead
from models.pipeline import Pipeline
from utils.normalize import normalize_data
from utils.plms_utils import load_plm


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute EVA PCA for NetLLM")
    parser.add_argument("--train-dataset", default="Jin2022")
    parser.add_argument("--plm-type", default="llama")
    parser.add_argument("--plm-size", default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-out", default="cuda")
    parser.add_argument("--device-mid", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--rank-budget", type=int, default=None)
    parser.add_argument("--min-rank", type=int, default=0)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--metric", choices=["raw", "ratio", "sum", "max"], default="ratio")
    parser.add_argument("--similarity-threshold", type=float, default=0.99)
    parser.add_argument("--min-batches", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--his-window", type=int, default=10)
    parser.add_argument("--fut-window", type=int, default=20)
    parser.add_argument("--trim-head", type=int, default=30)
    parser.add_argument("--trim-tail", type=int, default=60)
    parser.add_argument("--dataset-frequency", type=int, default=5)
    parser.add_argument("--sample-step", type=int, default=15)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--expected-target-modules", type=int, default=64)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("NetLLM's current viewport embedding path requires batch-size 1")
    if args.rank <= 0 or args.rho < 1:
        raise ValueError("rank must be positive and rho must be >= 1")
    max_components = round(args.rank * args.rho)
    configured_max = args.max_rank if args.max_rank is not None else max_components
    if configured_max > max_components:
        raise ValueError("max-rank cannot exceed round(rank * rho)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = os.path.join(cfg.plms_dir, args.plm_type, args.plm_size)
    plm, _, _ = load_plm(
        args.plm_type,
        model_path,
        plm_size=args.plm_size,
        device_input_side=args.device,
        device_output_side=args.device_out,
        device_middle_side=args.device_mid,
        torch_dtype=torch.float16 if args.fp16 else None,
    )
    if args.plm_type in ("opt", "gpt2") and args.plm_size != "large":
        plm = plm.to(args.device)
    for parameter in plm.parameters():
        parameter.requires_grad_(False)

    networking_head = NetworkingHead(
        input_dim=plm.hidden_size,
        output_dim=3,
        fut_window=args.fut_window,
    ).to(args.device_out)
    plm.set_networking_head(networking_head)
    pipeline = Pipeline(
        plm,
        fut_window=args.fut_window,
        device=args.device,
        embed_size=plm.hidden_size,
        frequency=args.dataset_frequency,
        multimodal_mode="none",
        dataset=args.train_dataset,
    )

    raw_train = create_dataset(
        args.train_dataset,
        his_window=args.his_window,
        fut_window=args.fut_window,
        trim_head=args.trim_head,
        trim_tail=args.trim_tail,
        include=["train"],
        frequency=args.dataset_frequency,
        step=args.sample_step,
    )[0]
    if args.limit_train_samples is not None:
        if args.limit_train_samples <= 0:
            raise ValueError("limit-train-samples must be positive")
        raw_train = torch.utils.data.Subset(
            raw_train, range(min(args.limit_train_samples, len(raw_train)))
        )
    generator = torch.Generator().manual_seed(args.seed)
    data_loader = DataLoader(
        raw_train,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        generator=generator,
    )

    expected_layers = (
        args.expected_target_modules // 2
        if args.plm_type == "llama" and args.expected_target_modules is not None
        else None
    )
    collector = EvaActivationCollector(
        pipeline.plm,
        target_modules=("q_proj", "v_proj"),
        max_components=max_components,
        similarity_threshold=args.similarity_threshold,
        expected_llama_layers=expected_layers,
    )
    module_count = len(collector.modules)
    if args.expected_target_modules is not None and module_count != args.expected_target_modules:
        raise ValueError(
            f"expected {args.expected_target_modules} EVA targets, found {module_count}"
        )
    rank_budget = args.rank_budget or module_count * args.rank

    def forward_batch(batch):
        history, future, video_user_info = batch
        history = normalize_data(history, args.train_dataset).to(args.device)
        future = normalize_data(future, args.train_dataset).to(args.device)
        pipeline.teaching_forcing(history, future, video_user_info)

    state = collector.collect(
        data_loader,
        forward_batch=forward_batch,
        rank_budget=rank_budget,
        metric=args.metric,
        min_rank=args.min_rank,
        max_rank=configured_max,
        min_batches=args.min_batches,
        max_batches=args.max_batches,
        require_convergence=not args.allow_unconverged,
    )
    metadata = save_eva_state(
        args.output_dir,
        state,
        metadata={
            "train_dataset": args.train_dataset,
            "plm_type": args.plm_type,
            "plm_size": args.plm_size,
            "model_path": os.path.abspath(model_path),
            "seed": args.seed,
            "rank": args.rank,
            "rho": args.rho,
            "min_rank": args.min_rank,
            "configured_max_rank": configured_max,
            "target_modules": ["q_proj", "v_proj"],
            "calibration_split": "train",
        },
    )
    print("EVA precomputation complete:", os.path.abspath(args.output_dir))
    print(metadata)


if __name__ == "__main__":
    main()
