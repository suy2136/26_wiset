"""Lightweight checks for independent master/LoRA/data RNG controls."""

import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.seed_utils import (  # noqa: E402
    isolated_seed,
    make_data_generator,
    resolve_experiment_seeds,
)


def _reset_outer(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _outer_draws():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def _loader_order(seed):
    dataset = TensorDataset(torch.arange(24))
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        generator=make_data_generator(seed),
    )
    return torch.cat([batch[0] for batch in loader]).tolist()


def main():
    assert resolve_experiment_seeds(7, None, None) == (7, 7, 7)
    assert resolve_experiment_seeds(7, 11, 13) == (7, 11, 13)
    print('[PASS] legacy --seed fallback and independent seed resolution')

    _reset_outer(101)
    expected_before = _outer_draws()
    expected_after = _outer_draws()

    _reset_outer(101)
    actual_before = _outer_draws()
    with isolated_seed(202, include_cuda=False):
        first = torch.rand(8)
    actual_after = _outer_draws()
    with isolated_seed(202, include_cuda=False):
        repeated = torch.rand(8)
    with isolated_seed(203, include_cuda=False):
        changed = torch.rand(8)

    assert actual_before == expected_before
    assert actual_after == expected_after
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    print('[PASS] LoRA seed scope is reproducible and restores outer RNG state')

    order_a = _loader_order(301)
    order_b = _loader_order(301)
    order_c = _loader_order(302)
    assert order_a == order_b
    assert order_a != order_c
    print('[PASS] DataLoader ordering depends only on its dedicated data seed')

    print('All seed-separation checks completed.')


if __name__ == '__main__':
    main()
