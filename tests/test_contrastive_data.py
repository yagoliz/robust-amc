"""Tests for contrastive data pipeline."""

import numpy as np
import pytest
import torch

from robust_amc.data.augmentations import MDADMCPipeline
from robust_amc.data.contrastive import (
    ContrastiveDataset,
    contrastive_collate_fn,
    get_contrastive_loaders,
)
from robust_amc.data.torchsig_dataset import TorchSigDataset
from robust_amc.data.transforms import Compose, PowerNormalize, ToTensor


def _create_mock_torchsig_data(path, n_per_class=40, signal_length=256, seed=42):
    """Create mock TorchSig data files (avoids TorchSig import issues)."""
    modulations = ["bpsk", "qpsk", "2fsk", "am-dsb", "am-usb"]
    n_samples = n_per_class * len(modulations)
    rng = np.random.default_rng(seed)

    data = (rng.standard_normal((n_samples, signal_length))
            + 1j * rng.standard_normal((n_samples, signal_length))).astype(np.complex64)
    # Repeat each modulation n_per_class times for balanced classes
    labels = np.array([m for m in modulations for _ in range(n_per_class)])
    # Use a small set of SNR values to avoid too-sparse stratification bins
    snr_values = np.array([-4.0, 0.0, 6.0, 12.0, 18.0])
    snrs = np.tile(snr_values, n_samples // len(snr_values) + 1)[:n_samples].astype(np.float32)

    np.save(path / "data.npy", data)
    np.save(path / "labels.npy", labels)
    np.save(path / "snrs.npy", snrs)
    return path


@pytest.fixture
def cache_dir(tmp_path):
    """Create mock TorchSig data with enough samples for stratified splitting."""
    return _create_mock_torchsig_data(tmp_path, n_per_class=100)


@pytest.fixture
def base_dataset(cache_dir):
    """Create a TorchSigDataset with no transform."""
    data = np.load(cache_dir / "data.npy")
    labels = np.load(cache_dir / "labels.npy", allow_pickle=True)
    snrs = np.load(cache_dir / "snrs.npy")
    return TorchSigDataset(data, labels, snrs, transform=None, crop_length=128, seed=42)


@pytest.fixture
def augmentation():
    return MDADMCPipeline(p=0.8, seed=123)


@pytest.fixture
def clean_transform():
    return Compose([PowerNormalize(), ToTensor()])


class TestContrastiveDataset:
    """Tests for ContrastiveDataset wrapper."""

    def test_returns_5_tuple(self, base_dataset, augmentation, clean_transform):
        ds = ContrastiveDataset(base_dataset, augmentation, clean_transform)
        sample = ds[0]
        assert len(sample) == 5, "Should return 5-tuple (x_i, x_j, x_orig, y, meta)"

    def test_correct_shapes(self, base_dataset, augmentation, clean_transform):
        ds = ContrastiveDataset(base_dataset, augmentation, clean_transform)
        x_i, x_j, x_orig, y, meta = ds[0]

        assert x_i.shape == (2, 128), f"x_i shape: {x_i.shape}"
        assert x_j.shape == (2, 128), f"x_j shape: {x_j.shape}"
        assert x_orig.shape == (2, 128), f"x_orig shape: {x_orig.shape}"
        assert isinstance(y, int)
        assert isinstance(meta, dict)
        assert "snr" in meta

    def test_all_tensors(self, base_dataset, augmentation, clean_transform):
        ds = ContrastiveDataset(base_dataset, augmentation, clean_transform)
        x_i, x_j, x_orig, _, _ = ds[0]

        assert isinstance(x_i, torch.Tensor)
        assert isinstance(x_j, torch.Tensor)
        assert isinstance(x_orig, torch.Tensor)

    def test_views_differ(self, base_dataset, clean_transform):
        """x_i and x_j should differ due to stochastic augmentation."""
        aug = MDADMCPipeline(p=1.0, seed=None)
        ds = ContrastiveDataset(base_dataset, aug, clean_transform)

        any_differ = False
        for idx in range(min(10, len(ds))):
            x_i, x_j, _, _, _ = ds[idx]
            if not torch.allclose(x_i, x_j, atol=1e-6):
                any_differ = True
                break

        assert any_differ, "x_i and x_j should differ for at least one sample"

    def test_x_orig_is_power_normalized(self, base_dataset, clean_transform):
        """x_orig should be power-normalized (unit power)."""
        aug = MDADMCPipeline(p=0.0)
        ds = ContrastiveDataset(base_dataset, aug, clean_transform)

        _, _, x_orig, _, _ = ds[0]
        power = torch.mean(x_orig[0] ** 2 + x_orig[1] ** 2)
        assert abs(power.item() - 1.0) < 0.01, f"x_orig power should be ~1.0, got {power.item()}"

    def test_length_matches_base(self, base_dataset, augmentation, clean_transform):
        ds = ContrastiveDataset(base_dataset, augmentation, clean_transform)
        assert len(ds) == len(base_dataset)


class TestContrastiveCollateFn:
    """Tests for contrastive_collate_fn."""

    def test_collate_shapes(self, base_dataset, augmentation, clean_transform):
        ds = ContrastiveDataset(base_dataset, augmentation, clean_transform)

        batch = [ds[i] for i in range(4)]
        x_i, x_j, x_orig, y, meta = contrastive_collate_fn(batch)

        assert x_i.shape == (4, 2, 128)
        assert x_j.shape == (4, 2, 128)
        assert x_orig.shape == (4, 2, 128)
        assert y.shape == (4,)
        assert y.dtype == torch.long
        assert meta["snr"].shape == (4,)
        assert meta["snr"].dtype == torch.float32


class TestGetContrastiveLoaders:
    """Tests for get_contrastive_loaders factory."""

    def test_returns_expected_keys(self, cache_dir):
        loaders = get_contrastive_loaders(
            str(cache_dir),
            batch_size=8,
            num_workers=0,
            seed=42,
        )
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders
        assert "family_names" in loaders
        assert len(loaders["family_names"]) == 5

    def test_train_produces_5_tuple(self, cache_dir):
        loaders = get_contrastive_loaders(
            str(cache_dir),
            batch_size=8,
            num_workers=0,
            seed=42,
        )
        batch = next(iter(loaders["train"]))
        assert len(batch) == 5, "Train batch should be 5-tuple"
        x_i, x_j, x_orig, y, meta = batch
        assert x_i.ndim == 3  # (batch, 2, 128)
        assert x_j.ndim == 3
        assert x_orig.ndim == 3

    def test_val_produces_5_tuple(self, cache_dir):
        loaders = get_contrastive_loaders(
            str(cache_dir),
            batch_size=8,
            num_workers=0,
            seed=42,
        )
        batch = next(iter(loaders["val"]))
        assert len(batch) == 5, "Val batch should be 5-tuple"

    def test_test_produces_3_tuple(self, cache_dir):
        loaders = get_contrastive_loaders(
            str(cache_dir),
            batch_size=8,
            num_workers=0,
            seed=42,
        )
        batch = next(iter(loaders["test"]))
        assert len(batch) == 3, "Test batch should be standard 3-tuple"
        x, y, meta = batch
        assert x.ndim == 3  # (batch, 2, 128)