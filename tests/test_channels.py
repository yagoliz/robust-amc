"""Tests for channel models."""

import numpy as np
import pytest
import torch

from robust_amc.data.channels import (
    AWGN,
    RayleighFading,
    RicianFading,
    TimeVaryingRayleigh,
)


class TestAWGN:
    def test_adds_noise(self):
        """AWGN should add noise to the signal."""
        x = np.zeros((2, 128), dtype=np.float32)
        x[0, :] = 1.0  # Unit amplitude I channel

        channel = AWGN(snr_db=10, seed=42)
        y = channel(x)

        # Output should differ from input
        assert not np.allclose(y, x)
        assert y.shape == x.shape

    def test_snr_approximately_correct(self):
        """Noise power should match target SNR."""
        # Create a known signal
        x = np.ones((2, 128), dtype=np.float32) * 0.5
        signal_power = np.mean(x[0] ** 2 + x[1] ** 2)

        target_snr_db = 10
        channel = AWGN(snr_db=target_snr_db, seed=42)

        # Average over multiple realizations
        noise_powers = []
        for i in range(100):
            channel_i = AWGN(snr_db=target_snr_db, seed=i)
            y = channel_i(x)
            noise = y - x
            noise_power = np.mean(noise[0] ** 2 + noise[1] ** 2)
            noise_powers.append(noise_power)

        avg_noise_power = np.mean(noise_powers)
        expected_noise_power = signal_power / (10 ** (target_snr_db / 10))

        # Allow 20% tolerance
        assert abs(avg_noise_power - expected_noise_power) / expected_noise_power < 0.2

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.ones(2, 128) * 0.5
        channel = AWGN(snr_db=10, seed=42)
        y = channel(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)

    def test_reproducibility(self):
        """Same seed should give same result."""
        x = np.random.randn(2, 128).astype(np.float32)

        channel1 = AWGN(snr_db=10, seed=42)
        channel2 = AWGN(snr_db=10, seed=42)

        y1 = channel1(x)
        y2 = channel2(x)

        np.testing.assert_allclose(y1, y2)


class TestRayleighFading:
    def test_applies_fading(self):
        """Rayleigh fading should modify the signal."""
        x = np.ones((2, 128), dtype=np.float32)

        channel = RayleighFading(seed=42)
        y = channel(x)

        # Output should differ from input
        assert not np.allclose(y, x)
        assert y.shape == x.shape

    def test_flat_fading(self):
        """Fading should be flat (same coefficient for all samples)."""
        x = np.ones((2, 128), dtype=np.float32)

        channel = RayleighFading(seed=42)
        y = channel(x)

        # All samples should have same amplitude (flat fading)
        amplitudes = np.sqrt(y[0] ** 2 + y[1] ** 2)
        assert np.allclose(amplitudes, amplitudes[0])

    def test_unit_average_power(self):
        """Average fading power should be approximately 1."""
        # Use unit power input: I=1, Q=0 so power = 1
        x = np.zeros((2, 128), dtype=np.float32)
        x[0, :] = 1.0

        fading_powers = []
        for seed in range(500):
            channel = RayleighFading(seed=seed)
            y = channel(x)
            # |h|^2 = |y|^2 / |x|^2 = |y|^2 for unit power input
            fading_power = y[0, 0] ** 2 + y[1, 0] ** 2
            fading_powers.append(fading_power)

        avg_power = np.mean(fading_powers)
        # Should be close to 1 (unit average power)
        assert abs(avg_power - 1.0) < 0.15

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.ones(2, 128)
        channel = RayleighFading(seed=42)
        y = channel(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)


class TestRicianFading:
    def test_applies_fading(self):
        """Rician fading should modify the signal."""
        x = np.ones((2, 128), dtype=np.float32)

        channel = RicianFading(k_factor=5, seed=42)
        y = channel(x)

        assert not np.allclose(y, x)
        assert y.shape == x.shape

    def test_high_k_factor_less_fading(self):
        """Higher K-factor should result in less fading variance."""
        x = np.ones((2, 128), dtype=np.float32)

        # Collect fading amplitudes for low and high K
        low_k_powers = []
        high_k_powers = []

        for seed in range(200):
            low_k = RicianFading(k_factor=0.1, seed=seed)
            high_k = RicianFading(k_factor=10, seed=seed)

            y_low = low_k(x)
            y_high = high_k(x)

            low_k_powers.append(y_low[0, 0] ** 2 + y_low[1, 0] ** 2)
            high_k_powers.append(y_high[0, 0] ** 2 + y_high[1, 0] ** 2)

        # High K should have lower variance
        assert np.var(high_k_powers) < np.var(low_k_powers)

    def test_k_zero_like_rayleigh(self):
        """K=0 should behave like Rayleigh fading."""
        # Use unit power input: I=1, Q=0 so power = 1
        x = np.zeros((2, 128), dtype=np.float32)
        x[0, :] = 1.0

        rician_powers = []
        for seed in range(500):
            channel = RicianFading(k_factor=0, seed=seed)
            y = channel(x)
            power = y[0, 0] ** 2 + y[1, 0] ** 2
            rician_powers.append(power)

        # Should have Rayleigh-like distribution (exponential power)
        # Mean should be ~1, variance should be ~1 for exponential
        assert abs(np.mean(rician_powers) - 1.0) < 0.2

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.ones(2, 128)
        channel = RicianFading(k_factor=5, seed=42)
        y = channel(x)

        assert isinstance(y, torch.Tensor)


class TestTimeVaryingRayleigh:
    def test_applies_fading(self):
        """Time-varying fading should modify the signal."""
        x = np.ones((2, 128), dtype=np.float32)

        channel = TimeVaryingRayleigh(doppler_hz=100, seed=42)
        y = channel(x)

        assert not np.allclose(y, x)
        assert y.shape == x.shape

    def test_time_varying(self):
        """Fading amplitude should vary over time."""
        x = np.ones((2, 128), dtype=np.float32)

        channel = TimeVaryingRayleigh(doppler_hz=1000, sample_rate=10000, seed=42)
        y = channel(x)

        amplitudes = np.sqrt(y[0] ** 2 + y[1] ** 2)
        # Should have variation (not flat)
        assert np.std(amplitudes) > 0.01

    def test_zero_doppler_flat(self):
        """Zero Doppler should give approximately flat fading."""
        x = np.ones((2, 128), dtype=np.float32)

        channel = TimeVaryingRayleigh(doppler_hz=0, seed=42)
        y = channel(x)

        amplitudes = np.sqrt(y[0] ** 2 + y[1] ** 2)
        # Should be approximately flat
        assert np.std(amplitudes) < 0.01

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.ones(2, 128)
        channel = TimeVaryingRayleigh(doppler_hz=100, seed=42)
        y = channel(x)

        assert isinstance(y, torch.Tensor)