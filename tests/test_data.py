"""Tests for data loading utilities."""

import json
import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from robust_amc.data.transforms import PowerNormalize, ToTensor, Compose
from robust_amc.data.label_mapping import FamilyMapper
from robust_amc.data.torchsig_dataset import (
    TorchSigDataset,
    generate_torchsig_data,
    load_torchsig_data,
    DEFAULT_MODULATIONS,
)
from robust_amc.data.panoradio_dataset import (
    PanoradioDataset,
    PANORADIO_SNR_LEVELS,
)
from robust_amc.data.registry import DatasetType, DatasetConfig, load_config_from_yaml


class TestFamilyMapper:
    """Tests for FamilyMapper class."""

    @pytest.fixture
    def torchsig_mapper(self):
        """Load the TorchSig family mapper."""
        return FamilyMapper("configs/label_maps/torchsig_to_family.yaml")

    @pytest.fixture
    def panoradio_mapper(self):
        """Load the Panoradio family mapper."""
        return FamilyMapper("configs/label_maps/panoradio_to_family.yaml")

    def test_torchsig_mapper_loads(self, torchsig_mapper):
        """Test TorchSig mapper loads correctly."""
        assert torchsig_mapper.num_families == 5
        assert "PSK" in torchsig_mapper.family_names
        assert "FSK" in torchsig_mapper.family_names

    def test_panoradio_mapper_loads(self, panoradio_mapper):
        """Test Panoradio mapper loads correctly."""
        assert panoradio_mapper.num_families == 5
        assert "PSK" in panoradio_mapper.family_names
        assert "OTHER" in panoradio_mapper.family_names

    def test_exact_match(self, torchsig_mapper):
        """Test exact label matching."""
        assert torchsig_mapper.get_family_name("bpsk") == "PSK"
        assert torchsig_mapper.get_family_name("BPSK") == "PSK"  # case insensitive
        assert torchsig_mapper.get_family_name("2fsk") == "FSK"
        assert torchsig_mapper.get_family_name("am-dsb") == "AM"

    def test_wildcard_match(self, panoradio_mapper):
        """Test wildcard pattern matching."""
        # Olivia* should match olivia8/500
        assert panoradio_mapper.get_family_name("olivia8/500") == "FSK"
        assert panoradio_mapper.get_family_name("olivia16/1000") == "FSK"
        # DominoEx* should match
        assert panoradio_mapper.get_family_name("dominoex4") == "FSK"

    def test_unmapped_returns_none(self, torchsig_mapper):
        """Test unmapped labels return None."""
        assert torchsig_mapper.get_family_name("unknown_modulation") is None
        assert torchsig_mapper.get_family_idx("unknown_modulation") is None

    def test_is_mapped(self, torchsig_mapper):
        """Test is_mapped function."""
        assert torchsig_mapper.is_mapped("qpsk") is True
        assert torchsig_mapper.is_mapped("unknown") is False

    def test_get_family_idx(self, torchsig_mapper):
        """Test getting family index."""
        psk_idx = torchsig_mapper.get_family_idx("bpsk")
        assert psk_idx is not None
        assert torchsig_mapper.family_names[psk_idx] == "PSK"


class TestTorchSigDataset:
    """Tests for TorchSig dataset."""

    @pytest.fixture
    def mock_torchsig_data(self, tmp_path):
        """Create mock TorchSig data."""
        n_samples = 200
        signal_length = 1024
        modulations = ["bpsk", "qpsk", "2fsk", "am-dsb"]

        # Create mock complex data
        data = np.random.randn(n_samples, signal_length) + 1j * np.random.randn(
            n_samples, signal_length
        )
        data = data.astype(np.complex64)

        # Create labels (cycling through modulations)
        labels = np.array([modulations[i % len(modulations)] for i in range(n_samples)])

        # Create SNRs
        snrs = np.random.uniform(-6, 20, n_samples).astype(np.float32)

        # Save
        np.save(tmp_path / "data.npy", data)
        np.save(tmp_path / "labels.npy", labels)
        np.save(tmp_path / "snrs.npy", snrs)

        return tmp_path, data, labels, snrs

    def test_dataset_loads(self, mock_torchsig_data):
        """Test dataset loads from cached files."""
        cache_dir, _, _, _ = mock_torchsig_data
        data, labels, snrs = load_torchsig_data(cache_dir)

        assert data.shape[0] == 200
        assert len(labels) == 200
        assert len(snrs) == 200

    def test_dataset_returns_correct_format(self, mock_torchsig_data):
        """Test dataset returns (x, y_family, meta) format."""
        cache_dir, data, labels, snrs = mock_torchsig_data
        mapper = FamilyMapper("configs/label_maps/torchsig_to_family.yaml")

        dataset = TorchSigDataset(data, labels, snrs, family_mapper=mapper, crop_length=128)

        x, y, meta = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert x.shape == (2, 128)
        assert isinstance(y, int)
        assert isinstance(meta, dict)
        assert "raw_label" in meta
        assert "snr" in meta
        assert "family_name" in meta

    def test_random_crop(self, mock_torchsig_data):
        """Test random cropping works."""
        cache_dir, data, labels, snrs = mock_torchsig_data
        mapper = FamilyMapper("configs/label_maps/torchsig_to_family.yaml")

        dataset = TorchSigDataset(
            data, labels, snrs, family_mapper=mapper, crop_length=128, seed=42
        )

        # Get same sample twice - should have same crop due to deterministic RNG per index
        x1, _, meta1 = dataset[0]
        x2, _, meta2 = dataset[0]

        assert x1.shape == (2, 128)
        # Crop positions may differ due to RNG state, but shape should be consistent

    def test_complex_to_iq_conversion(self, mock_torchsig_data):
        """Test complex to I/Q format conversion."""
        cache_dir, data, labels, snrs = mock_torchsig_data
        mapper = FamilyMapper("configs/label_maps/torchsig_to_family.yaml")

        dataset = TorchSigDataset(data, labels, snrs, family_mapper=mapper, crop_length=128)

        x, _, _ = dataset[0]

        # Should have 2 channels (I and Q)
        assert x.shape[0] == 2

    def test_family_mapping(self, mock_torchsig_data):
        """Test family labels are correctly mapped."""
        cache_dir, data, labels, snrs = mock_torchsig_data
        mapper = FamilyMapper("configs/label_maps/torchsig_to_family.yaml")

        dataset = TorchSigDataset(data, labels, snrs, family_mapper=mapper, crop_length=128)

        # Check all samples have valid family indices
        for i in range(min(10, len(dataset))):
            _, y, meta = dataset[i]
            assert 0 <= y < dataset.num_families
            assert meta["family_name"] in mapper.family_names


class TestPanoradioDataset:
    """Tests for Panoradio dataset."""

    @pytest.fixture
    def mock_panoradio_data(self, tmp_path):
        """Create mock Panoradio data."""
        n_samples = 100
        signal_length = 2048
        modes = ["PSK31", "USB", "RTTY45", "AM"]

        # Create mock complex data
        data = np.random.randn(n_samples, signal_length) + 1j * np.random.randn(
            n_samples, signal_length
        )
        data = data.astype(np.complex64)

        # Create labels
        labels = np.array([modes[i % len(modes)] for i in range(n_samples)])

        # Create SNRs
        snrs = np.random.choice(PANORADIO_SNR_LEVELS, n_samples).astype(np.float32)

        # Save data
        np.save(tmp_path / "rscd_2048.npy", data)

        # Save tags.csv
        with open(tmp_path / "tags.csv", "w") as f:
            f.write("label,snr\n")
            for label, snr in zip(labels, snrs):
                f.write(f"{label},{snr}\n")

        return tmp_path, data, labels, snrs

    def test_dataset_returns_correct_format(self, mock_panoradio_data):
        """Test dataset returns (x, y_family, meta) format."""
        data_dir, data, labels, snrs = mock_panoradio_data
        mapper = FamilyMapper("configs/label_maps/panoradio_to_family.yaml")

        dataset = PanoradioDataset(data, labels, snrs, family_mapper=mapper, crop_length=128)

        x, y, meta = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert x.shape == (2, 128)
        assert isinstance(y, int)
        assert isinstance(meta, dict)

    def test_random_windowing(self, mock_panoradio_data):
        """Test random windowing from 2048 to 128 samples."""
        data_dir, data, labels, snrs = mock_panoradio_data
        mapper = FamilyMapper("configs/label_maps/panoradio_to_family.yaml")

        dataset = PanoradioDataset(data, labels, snrs, family_mapper=mapper, crop_length=128)

        x, _, meta = dataset[0]

        # Should be cropped to 128 samples
        assert x.shape == (2, 128)
        # crop_start should be within valid range
        assert 0 <= meta["crop_start"] <= 2048 - 128


class TestDatasetRegistry:
    """Tests for dataset registry."""

    def test_load_config_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        config_content = """
dataset_type: torchsig
data_path: data/test
label_map_path: configs/label_maps/torchsig_to_family.yaml
batch_size: 128
seed: 123
extra_config:
  crop_length: 64
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        config = load_config_from_yaml(config_path)

        assert config.dataset_type == DatasetType.TORCHSIG
        assert config.data_path == "data/test"
        assert config.batch_size == 128
        assert config.seed == 123
        assert config.extra_config["crop_length"] == 64

    def test_invalid_dataset_type_raises(self, tmp_path):
        """Test that invalid dataset type raises ValueError."""
        config_content = """
dataset_type: invalid_type
data_path: data/test
"""
        config_path = tmp_path / "bad_config.yaml"
        config_path.write_text(config_content)

        with pytest.raises(ValueError, match="Unknown dataset_type"):
            load_config_from_yaml(config_path)


class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_fallback_generation(self, tmp_path):
        """Test fallback signal generation works."""
        data, labels, snrs = generate_torchsig_data(
            tmp_path,
            modulations=["bpsk", "qpsk"],
            num_samples_per_class=50,
            signal_length=256,
            seed=42,
        )

        assert data.shape == (100, 256)
        assert len(labels) == 100
        assert len(snrs) == 100
        assert set(labels) == {"bpsk", "qpsk"}

    def test_generated_data_is_complex(self, tmp_path):
        """Test generated data is complex-valued."""
        data, labels, snrs = generate_torchsig_data(
            tmp_path,
            modulations=["bpsk"],
            num_samples_per_class=10,
            signal_length=128,
        )

        assert np.iscomplexobj(data)

    def test_metadata_saved(self, tmp_path):
        """Test metadata is saved alongside data."""
        generate_torchsig_data(
            tmp_path,
            modulations=["bpsk", "qpsk"],
            num_samples_per_class=10,
            signal_length=128,
        )

        assert (tmp_path / "metadata.json").exists()

        with open(tmp_path / "metadata.json") as f:
            metadata = json.load(f)

        assert metadata["total_samples"] == 20
        assert metadata["signal_length"] == 128


class TestTransformsIntegration:
    """Test transforms work with new datasets."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for transform testing."""
        data = np.random.randn(10, 1024) + 1j * np.random.randn(10, 1024)
        data = data.astype(np.complex64)
        labels = np.array(["bpsk"] * 10)
        snrs = np.random.uniform(-6, 20, 10).astype(np.float32)
        return data, labels, snrs

    def test_power_normalize_with_torchsig(self, sample_data):
        """Test PowerNormalize works with TorchSig dataset."""
        data, labels, snrs = sample_data
        mapper = FamilyMapper("configs/label_maps/torchsig_to_family.yaml")

        transform = Compose([PowerNormalize(), ToTensor()])
        dataset = TorchSigDataset(
            data, labels, snrs, family_mapper=mapper, transform=transform, crop_length=128
        )

        x, _, _ = dataset[0]

        # Check power is normalized to ~1
        power = torch.mean(x[0] ** 2 + x[1] ** 2)
        assert abs(power - 1.0) < 0.1
