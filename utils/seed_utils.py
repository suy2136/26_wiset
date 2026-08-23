"""Reproducible, independently controlled RNG scopes for NetLLM experiments."""

from contextlib import contextmanager
import random

import numpy as np
import torch


def resolve_experiment_seeds(seed, lora_seed=None, data_seed=None):
    """Resolve optional component seeds while preserving the legacy single seed."""
    seed = int(seed)
    resolved_lora = seed if lora_seed is None else int(lora_seed)
    resolved_data = seed if data_seed is None else int(data_seed)
    if min(seed, resolved_lora, resolved_data) < 0:
        raise ValueError('seed, lora_seed, and data_seed must be non-negative')
    return seed, resolved_lora, resolved_data


@contextmanager
def isolated_seed(seed, include_cuda=True):
    """Temporarily seed Python, NumPy, and torch without advancing outer RNGs."""
    seed = int(seed)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = (
        list(range(torch.cuda.device_count()))
        if include_cuda and torch.cuda.is_available()
        else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def make_data_generator(seed, offset=0):
    """Create an independent CPU generator for DataLoader sampling."""
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def seed_data_worker(worker_id):
    """Seed Python/NumPy inside a DataLoader worker from its torch worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
