"""Tests for reproducibility utilities."""

import random

import numpy as np
import pytest
import torch

from robust_amc.utils import seed_worker, set_seed


class TestSetSeed:
    """Tests for set_seed function."""

    def test_python_random_reproducible(self):
        """Test that Python random is reproducible with same seed."""
        set_seed(42)
        values1 = [random.random() for _ in range(10)]

        set_seed(42)
        values2 = [random.random() for _ in range(10)]

        assert values1 == values2

    def test_numpy_random_reproducible(self):
        """Test that NumPy random is reproducible with same seed."""
        set_seed(42)
        arr1 = np.random.randn(10)

        set_seed(42)
        arr2 = np.random.randn(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_torch_random_reproducible(self):
        """Test that PyTorch random is reproducible with same seed."""
        set_seed(42)
        tensor1 = torch.randn(10)

        set_seed(42)
        tensor2 = torch.randn(10)

        torch.testing.assert_close(tensor1, tensor2)

    def test_torch_model_init_reproducible(self):
        """Test that model initialization is reproducible with same seed."""
        set_seed(42)
        model1 = torch.nn.Linear(10, 5)
        weights1 = model1.weight.clone()

        set_seed(42)
        model2 = torch.nn.Linear(10, 5)
        weights2 = model2.weight.clone()

        torch.testing.assert_close(weights1, weights2)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        values1 = [random.random() for _ in range(10)]

        set_seed(123)
        values2 = [random.random() for _ in range(10)]

        assert values1 != values2

    def test_all_generators_seeded_together(self):
        """Test that all generators are seeded consistently."""
        set_seed(42)
        py_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        set_seed(42)
        assert random.random() == py_val
        assert np.random.rand() == np_val
        assert torch.rand(1).item() == torch_val


class TestSeedWorker:
    """Tests for seed_worker function."""

    def test_seed_worker_sets_seeds(self):
        """Test that seed_worker sets numpy and random seeds."""
        # Simulate what DataLoader does
        torch.manual_seed(42)

        # Call seed_worker as DataLoader would
        seed_worker(0)

        # Capture values
        np_val = np.random.rand()
        py_val = random.random()

        # Reset and repeat
        torch.manual_seed(42)
        seed_worker(0)

        assert np.random.rand() == np_val
        assert random.random() == py_val

    def test_different_workers_different_seeds(self):
        """Test that different torch initial seeds produce different worker seeds.

        In practice, DataLoader sets different torch seeds for each worker
        before calling worker_init_fn. This test simulates that behavior.
        """
        # Simulate worker 0 with base seed 42
        torch.manual_seed(42)
        seed_worker(0)
        values_worker0 = [np.random.rand() for _ in range(5)]

        # Simulate worker 1 with different base seed (as DataLoader would do)
        torch.manual_seed(43)
        seed_worker(1)
        values_worker1 = [np.random.rand() for _ in range(5)]

        # Worker seeds should be different when torch seeds differ
        assert values_worker0 != values_worker1


class TestDeterministicMode:
    """Tests for deterministic mode."""

    def test_deterministic_flag(self):
        """Test that deterministic flag can be set without error."""
        # Just test that it doesn't raise an exception
        # Actual deterministic behavior depends on operations used
        set_seed(42, deterministic=True)

        # Verify basic operations still work
        _ = torch.randn(10)
        _ = np.random.randn(10)