"""Independent RNG controls for reproducible ABR experiments."""

from contextlib import contextmanager
import random

import numpy as np
import torch


def resolve_experiment_seeds(seed, lora_seed=None, data_seed=None):
    """Resolve optional component seeds while preserving legacy --seed."""
    seed = int(seed)
    lora_seed = seed if lora_seed is None else int(lora_seed)
    data_seed = seed if data_seed is None else int(data_seed)
    if min(seed, lora_seed, data_seed) < 0:
        raise ValueError('seed, lora_seed, and data_seed must be non-negative')
    return seed, lora_seed, data_seed


@contextmanager
def isolated_seed(seed, include_cuda=True):
    """Temporarily seed adapter initialization without advancing master RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = (
        list(range(torch.cuda.device_count()))
        if include_cuda and torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if cuda_devices:
                torch.cuda.manual_seed_all(int(seed))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def make_data_generator(seed):
    """Return a DataLoader generator independent of model/dropout RNG state."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator

