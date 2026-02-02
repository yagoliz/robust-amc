"""Tests for hardware impairment models."""

import numpy as np
import pytest
import torch

from robust_amc.data.impairments import (
    CarrierFrequencyOffset,
    IQImbalance,
    DCOffset,
    PhaseNoise,
    SampleRateOffset,
)


class TestCarrierFrequencyOffset:
    def test_applies_rotation(self):
        """CFO should rotate the signal phase over time."""
        x = np.ones((2, 128), dtype=np.float32)
        x[1, :] = 0  # Pure I signal

        cfo = CarrierFrequencyOffset(delta_f=1000, sample_rate=128000)
        y = cfo(x)

        # Phase should change over time
        phases = np.arctan2(y[1], y[0])
        phases_unwrapped = np.unwrap(phases)
        phase_diff = np.diff(phases_unwrapped)

        # Phase should increase (or decrease) monotonically
        assert np.std(phase_diff) < 0.01  # Consistent phase change

    def test_zero_offset_no_change(self):
        """Zero frequency offset should not change the signal (except initial phase)."""
        x = np.random.randn(2, 128).astype(np.float32)

        cfo = CarrierFrequencyOffset(delta_f=0, sample_rate=1e6, initial_phase=0)
        y = cfo(x)

        np.testing.assert_allclose(y, x, rtol=1e-5)

    def test_rotation_rate(self):
        """Phase rotation rate should match frequency offset."""
        x = np.ones((2, 128), dtype=np.float32)
        x[1, :] = 0

        delta_f = 1000  # Hz
        sample_rate = 128000  # Hz

        cfo = CarrierFrequencyOffset(
            delta_f=delta_f, sample_rate=sample_rate, initial_phase=0
        )
        y = cfo(x)

        phases = np.arctan2(y[1], y[0])
        # Unwrap phase to handle discontinuities
        phases_unwrapped = np.unwrap(phases)

        # Phase change per sample should be 2*pi*delta_f/sample_rate
        expected_phase_rate = 2 * np.pi * delta_f / sample_rate
        actual_phase_rate = np.mean(np.diff(phases_unwrapped))

        assert abs(actual_phase_rate - expected_phase_rate) < 0.01

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        cfo = CarrierFrequencyOffset(delta_f=1000, sample_rate=128000)
        y = cfo(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)


class TestIQImbalance:
    def test_zero_imbalance_no_change(self):
        """Zero imbalance should not change the signal."""
        x = np.random.randn(2, 128).astype(np.float32)

        iq_imb = IQImbalance(amplitude_imbalance_db=0, phase_imbalance_deg=0)
        y = iq_imb(x)

        np.testing.assert_allclose(y, x, rtol=1e-5)

    def test_amplitude_imbalance(self):
        """Amplitude imbalance should scale Q channel."""
        x = np.ones((2, 128), dtype=np.float32)

        iq_imb = IQImbalance(amplitude_imbalance_db=3, phase_imbalance_deg=0)
        y = iq_imb(x)

        # Q channel should be scaled by 10^(3/20) ≈ 1.41
        expected_q_scale = 10 ** (3 / 20)
        assert abs(y[1, 0] / x[1, 0] - expected_q_scale) < 0.01
        # I channel should be unchanged
        np.testing.assert_allclose(y[0], x[0])

    def test_phase_imbalance(self):
        """Phase imbalance should mix I into Q."""
        x = np.zeros((2, 128), dtype=np.float32)
        x[0, :] = 1.0  # Pure I signal

        iq_imb = IQImbalance(amplitude_imbalance_db=0, phase_imbalance_deg=5)
        y = iq_imb(x)

        # Q should have some leakage from I
        phi = np.deg2rad(5)
        expected_q = np.sin(phi) * x[0, 0]
        assert abs(y[1, 0] - expected_q) < 0.01

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        iq_imb = IQImbalance(amplitude_imbalance_db=1, phase_imbalance_deg=2)
        y = iq_imb(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)


class TestDCOffset:
    def test_adds_offset(self):
        """DC offset should add constant to channels."""
        x = np.zeros((2, 128), dtype=np.float32)

        dc = DCOffset(dc_i=0.5, dc_q=-0.3)
        y = dc(x)

        np.testing.assert_allclose(y[0], 0.5)
        np.testing.assert_allclose(y[1], -0.3)

    def test_zero_offset_no_change(self):
        """Zero offset should not change the signal."""
        x = np.random.randn(2, 128).astype(np.float32)

        dc = DCOffset(dc_i=0, dc_q=0)
        y = dc(x)

        np.testing.assert_allclose(y, x)

    def test_relative_offset(self):
        """Relative offset should scale with signal amplitude."""
        x = np.ones((2, 128), dtype=np.float32) * 2

        dc = DCOffset(dc_i=0.1, dc_q=0.1, relative=True)
        y = dc(x)

        # Signal power = 2^2 + 2^2 = 8, sqrt = 2.83
        signal_rms = np.sqrt(np.mean(x[0] ** 2 + x[1] ** 2))
        expected_offset = 0.1 * signal_rms

        assert abs(y[0, 0] - (x[0, 0] + expected_offset)) < 0.01

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        dc = DCOffset(dc_i=0.1, dc_q=0.1)
        y = dc(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)


class TestPhaseNoise:
    def test_applies_phase_variation(self):
        """Phase noise should cause phase to vary."""
        x = np.ones((2, 128), dtype=np.float32)
        x[1, :] = 0  # Pure I signal

        pn = PhaseNoise(std_per_sample=0.1, seed=42)
        y = pn(x)

        # Output phase should vary
        phases = np.arctan2(y[1], y[0])
        assert np.std(phases) > 0.01

    def test_preserves_amplitude(self):
        """Phase noise should preserve signal amplitude."""
        x = np.ones((2, 128), dtype=np.float32)

        pn = PhaseNoise(std_per_sample=0.1, seed=42)
        y = pn(x)

        # Amplitude should be preserved
        input_amp = np.sqrt(x[0] ** 2 + x[1] ** 2)
        output_amp = np.sqrt(y[0] ** 2 + y[1] ** 2)

        np.testing.assert_allclose(input_amp, output_amp, rtol=1e-5)

    def test_zero_noise_no_change(self):
        """Zero phase noise should not change the signal."""
        x = np.random.randn(2, 128).astype(np.float32)

        pn = PhaseNoise(std_per_sample=0, seed=42)
        y = pn(x)

        np.testing.assert_allclose(y, x, rtol=1e-5)

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        pn = PhaseNoise(std_per_sample=0.1, seed=42)
        y = pn(x)

        assert isinstance(y, torch.Tensor)


class TestSampleRateOffset:
    def test_zero_offset_no_change(self):
        """Zero SRO should not change the signal."""
        x = np.random.randn(2, 128).astype(np.float32)

        sro = SampleRateOffset(ppm=0)
        y = sro(x)

        np.testing.assert_allclose(y, x, rtol=1e-5)

    def test_positive_ppm_compresses(self):
        """Positive PPM should compress the signal (faster receiver)."""
        # Create a signal with known frequency content
        t = np.linspace(0, 1, 128)
        x = np.stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t)]).astype(
            np.float32
        )

        sro = SampleRateOffset(ppm=10000)  # Large offset for visibility
        y = sro(x)

        # Signal should be different due to resampling
        assert not np.allclose(y, x)

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        sro = SampleRateOffset(ppm=100)
        y = sro(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        x = np.random.randn(2, 256).astype(np.float32)

        sro = SampleRateOffset(ppm=50)
        y = sro(x)

        assert y.shape == x.shape