"""Precompute task-specific EVA activation PCA for ABR NetLLM.

The script calibrates the frozen base Llama on deterministic samples from the
same ABR experience pool used for training.  It only writes EVA state and
diagnostics; it never modifies an existing LoRA/NBS checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import random
import sys

ABR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ABR_ROOT.parent
for path in (str(PROJECT_ROOT), str(ABR_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch

from config import cfg
from models.eva_initializer import EvaActivationCollector, save_eva_state
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.utils.plm_utils import load_plm
from plm_special.utils.utils import process_batch, set_random_seed
from baseline_special.utils.constants import BITRATE_LEVELS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plm-dir', type=Path, required=True)
    parser.add_argument('--exp-pool-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--rank-budget', type=int, default=1536)
    parser.add_argument('--min-rank', type=int, default=2)
    parser.add_argument('--max-rank', type=int, default=32)
    parser.add_argument(
        '--metric', choices=('raw', 'ratio', 'sum', 'max'), default='ratio'
    )
    parser.add_argument('--similarity-threshold', type=float, default=0.99)
    parser.add_argument('--min-batches', type=int, default=2)
    parser.add_argument('--max-batches', type=int, default=128)
    parser.add_argument('--allow-unconverged', action='store_true')
    parser.add_argument('--w', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--scale', type=int, default=1000)
    parser.add_argument('--sample-step', type=int, default=None)
    parser.add_argument('--state-feature-dim', type=int, default=256)
    parser.add_argument('--expected-target-modules', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.plm_dir.joinpath('config.json').is_file():
        raise FileNotFoundError(f'base model not found: {args.plm_dir}')
    if not args.exp_pool_path.is_file():
        raise FileNotFoundError(
            f'experience pool not found: {args.exp_pool_path}'
        )
    if not 0 <= args.min_rank <= args.max_rank:
        raise ValueError('require 0 <= min-rank <= max-rank')
    minimum = args.min_rank * args.expected_target_modules
    maximum = args.max_rank * args.expected_target_modules
    if not minimum <= args.rank_budget <= maximum:
        raise ValueError(
            f'rank budget {args.rank_budget} is outside [{minimum}, {maximum}]'
        )
    if args.min_batches <= 0 or args.max_batches < args.min_batches:
        raise ValueError('require 0 < min-batches <= max-batches')

    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with args.exp_pool_path.open('rb') as stream:
        exp_pool = pickle.load(stream)
    dataset = ExperienceDataset(
        exp_pool,
        gamma=args.gamma,
        scale=args.scale,
        max_length=args.w,
        sample_step=args.sample_step,
    )
    if not len(dataset):
        raise RuntimeError('ABR experience dataset is empty')

    plm, *_ = load_plm(
        'llama', str(args.plm_dir.resolve()),
        device_input_side=args.device,
        device_output_side=args.device,
        torch_dtype=torch.float16 if args.fp16 else None,
    )
    for parameter in plm.parameters():
        parameter.requires_grad_(False)
    state_encoder = EncoderNetwork(
        embed_dim=args.state_feature_dim
    ).to(args.device)
    policy = OfflineRLPolicy(
        state_feature_dim=args.state_feature_dim,
        bitrate_levels=BITRATE_LEVELS,
        state_encoder=state_encoder,
        plm=plm,
        plm_embed_size=cfg.plm_embed_sizes['llama']['base'],
        max_length=args.w,
        max_ep_len=dataset.exp_dataset_info['max_timestep'] + 1,
        device=args.device,
        device_out=args.device,
        temporal_selector=None,
        token_selector=None,
        speculative_draft_steps=0,
    ).eval()

    collector = EvaActivationCollector(
        policy.plm,
        target_modules=('q_proj', 'v_proj'),
        max_components=args.max_rank,
        similarity_threshold=args.similarity_threshold,
        expected_llama_layers=args.expected_target_modules // 2,
    )
    if len(collector.modules) != args.expected_target_modules:
        raise RuntimeError(
            f'expected {args.expected_target_modules} EVA modules, '
            f'found {len(collector.modules)}'
        )

    generator = np.random.default_rng(args.seed)
    order = generator.permutation(len(dataset))[
        :min(len(dataset), args.max_batches)
    ]
    calibration_batches = [dataset[int(index)] for index in order]

    def forward_batch(batch):
        states, actions, returns, timesteps, _ = process_batch(
            batch, device=args.device
        )
        policy(states, actions, returns, timesteps)

    state = collector.collect(
        calibration_batches,
        forward_batch=forward_batch,
        rank_budget=args.rank_budget,
        metric=args.metric,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        min_batches=args.min_batches,
        max_batches=min(args.max_batches, len(calibration_batches)),
        require_convergence=not args.allow_unconverged,
    )
    metadata = save_eva_state(
        args.output_dir,
        state,
        metadata={
            'task': 'adaptive_bitrate_streaming',
            'base_model': os.path.abspath(args.plm_dir),
            'experience_pool': os.path.abspath(args.exp_pool_path),
            'calibration_split': 'training_experience_pool',
            'seed': args.seed,
            'w': args.w,
            'gamma': args.gamma,
            'scale': args.scale,
            'min_rank': args.min_rank,
            'configured_max_rank': args.max_rank,
            'target_modules': ['q_proj', 'v_proj'],
            'fp16_base': bool(args.fp16),
        },
    )
    print('ABR EVA precomputation complete:', args.output_dir.resolve())
    print(metadata)


if __name__ == '__main__':
    main()
