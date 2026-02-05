"""Reproducibility utilities for seeding random number generators."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds for all random number generators.

    This function sets seeds for Python's random module, NumPy, and PyTorch
    to enable reproducible results across training runs.

    Args:
        seed: The seed value to use for all random number generators.
        deterministic: If True, also enable PyTorch deterministic mode.
            This may impact performance but ensures fully reproducible results.
            Note: Some operations may not have deterministic implementations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # MPS (Apple Silicon) seeding
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't have manual_seed, but torch.manual_seed covers it
        pass

    if deterministic:
        torch.use_deterministic_algorithms(True)
        # Required for CUDA deterministic operations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:
    """Seed function for DataLoader workers.

    Use this with DataLoader's worker_init_fn parameter to ensure
    each worker has a unique but reproducible seed.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker)

    Args:
        worker_id: The worker ID (provided automatically by DataLoader).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)