"""Tests for data loading utilities."""

import numpy as np
import pytest
import torch

from robust_amc.data.radioml_loader import (
    RadioMLDataset,
    MODULATION_CLASSES,
    SNR_LEVELS,
    stratified_split,
)
from robust_amc.data.transforms import PowerNormalize, ToTensor, Compose


class TestRadioMLDataset:
    """Tests for RadioMLDataset class using synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic dataset mimicking RadioML structure."""
        n_samples = 1000
        n_classes = len(MODULATION_CLASSES)
        seq_len = 128

        # Create random I/Q data
        data = np.random.randn(n_samples, 2, seq_len).astype(np.float32)

        # Create labels (balanced across classes)
        labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
        labels = np.concatenate([labels, np.zeros(n_samples - len(labels), dtype=int)])
        np.random.shuffle(labels)

        # Create SNRs (random from valid levels)
        snrs = np.random.choice(SNR_LEVELS, size=n_samples).astype(np.float32)

        return data, labels, snrs

    def test_dataset_length(self, synthetic_data):
        data, labels, snrs = synthetic_data
        dataset = RadioMLDataset(data, labels, snrs)
        assert len(dataset) == len(labels)

    def test_dataset_getitem(self, synthetic_data):
        data, labels, snrs = synthetic_data
        dataset = RadioMLDataset(data, labels, snrs)

        x, y, s = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert x.shape == (2, 128)
        assert isinstance(y, (int, np.integer))
        assert isinstance(s, (float, np.floating))

    def test_dataset_with_transform(self, synthetic_data):
        data, labels, snrs = synthetic_data
        transform = Compose([PowerNormalize(), ToTensor()])
        dataset = RadioMLDataset(data, labels, snrs, transform=transform)

        x, y, s = dataset[0]
        assert isinstance(x, torch.Tensor)

        # Check power normalization worked
        power = torch.mean(x[0] ** 2 + x[1] ** 2)
        assert abs(power - 1.0) < 0.01

    def test_num_classes(self, synthetic_data):
        data, labels, snrs = synthetic_data
        dataset = RadioMLDataset(data, labels, snrs)
        assert dataset.num_classes == len(MODULATION_CLASSES)

    def test_class_names(self, synthetic_data):
        data, labels, snrs = synthetic_data
        dataset = RadioMLDataset(data, labels, snrs)
        assert dataset.class_names == MODULATION_CLASSES


class TestStratifiedSplit:
    """Tests for stratified splitting functionality."""

    @pytest.fixture
    def balanced_data(self):
        """Create balanced synthetic dataset."""
        n_per_class_snr = 10
        n_classes = 5
        n_snrs = 5
        seq_len = 128

        data_list = []
        labels_list = []
        snrs_list = []

        for cls in range(n_classes):
            for snr_idx, snr in enumerate(SNR_LEVELS[:n_snrs]):
                samples = np.random.randn(n_per_class_snr, 2, seq_len).astype(np.float32)
                data_list.append(samples)
                labels_list.extend([cls] * n_per_class_snr)
                snrs_list.extend([snr] * n_per_class_snr)

        data = np.vstack(data_list)
        labels = np.array(labels_list)
        snrs = np.array(snrs_list)

        return data, labels, snrs

    def test_split_ratios(self, balanced_data):
        data, labels, snrs = balanced_data
        splits = stratified_split(
            data, labels, snrs,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        total = len(labels)
        train_ratio = len(splits["train"][1]) / total
        val_ratio = len(splits["val"][1]) / total
        test_ratio = len(splits["test"][1]) / total

        assert abs(train_ratio - 0.6) < 0.05
        assert abs(val_ratio - 0.2) < 0.05
        assert abs(test_ratio - 0.2) < 0.05

    def test_no_overlap(self, balanced_data):
        data, labels, snrs = balanced_data
        splits = stratified_split(data, labels, snrs)

        # Get indices by comparing data arrays
        train_set = set(map(tuple, splits["train"][0].reshape(len(splits["train"][0]), -1)))
        val_set = set(map(tuple, splits["val"][0].reshape(len(splits["val"][0]), -1)))
        test_set = set(map(tuple, splits["test"][0].reshape(len(splits["test"][0]), -1)))

        # Check no overlap
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_reproducibility(self, balanced_data):
        data, labels, snrs = balanced_data

        splits1 = stratified_split(data, labels, snrs, random_state=42)
        splits2 = stratified_split(data, labels, snrs, random_state=42)

        np.testing.assert_array_equal(splits1["train"][1], splits2["train"][1])
        np.testing.assert_array_equal(splits1["val"][1], splits2["val"][1])
        np.testing.assert_array_equal(splits1["test"][1], splits2["test"][1])

    def test_different_seeds_different_splits(self, balanced_data):
        data, labels, snrs = balanced_data

        splits1 = stratified_split(data, labels, snrs, random_state=42)
        splits2 = stratified_split(data, labels, snrs, random_state=123)

        # Splits should be different
        assert not np.array_equal(splits1["train"][1], splits2["train"][1])


class TestModulationClasses:
    """Tests for modulation class constants."""

    def test_modulation_count(self):
        assert len(MODULATION_CLASSES) == 11

    def test_known_modulations(self):
        expected = {"QPSK", "BPSK", "8PSK", "QAM16", "QAM64", "WBFM"}
        assert expected.issubset(set(MODULATION_CLASSES))

    def test_snr_levels(self):
        assert SNR_LEVELS[0] == -20
        assert SNR_LEVELS[-1] == 18
        assert len(SNR_LEVELS) == 20
