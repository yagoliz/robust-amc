"""Tests for data transforms."""

import numpy as np
import pytest
import torch

from robust_amc.data.transforms import (
    Compose,
    ToTensor,
    PowerNormalize,
    Normalize,
    ComplexToIQ,
    IQToComplex,
)


class TestToTensor:
    def test_numpy_to_tensor(self):
        x = np.random.randn(2, 128).astype(np.float32)
        transform = ToTensor()
        result = transform(x)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (2, 128)

    def test_tensor_passthrough(self):
        x = torch.randn(2, 128)
        transform = ToTensor()
        result = transform(x)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32


class TestPowerNormalize:
    def test_unit_power(self):
        x = np.random.randn(2, 128).astype(np.float32) * 5  # Amplified signal
        transform = PowerNormalize()
        result = transform(x)

        # Check power is approximately 1
        power = np.mean(result[0] ** 2 + result[1] ** 2)
        assert abs(power - 1.0) < 0.01

    def test_preserves_phase(self):
        # Create signal with known phase
        t = np.linspace(0, 2 * np.pi, 128)
        x = np.stack([np.cos(t), np.sin(t)]).astype(np.float32) * 3

        transform = PowerNormalize()
        result = transform(x)

        # Phase should be preserved (atan2 ratio should be same)
        original_phase = np.arctan2(x[1], x[0])
        result_phase = np.arctan2(result[1], result[0])

        np.testing.assert_allclose(original_phase, result_phase, rtol=1e-5)

    def test_tensor_input(self):
        x = torch.randn(2, 128) * 5
        transform = PowerNormalize()
        result = transform(x)

        assert isinstance(result, torch.Tensor)
        power = torch.mean(result[0] ** 2 + result[1] ** 2)
        assert abs(power - 1.0) < 0.01


class TestNormalize:
    def test_zero_mean_unit_var(self):
        x = np.random.randn(2, 128).astype(np.float32) * 5 + 3
        transform = Normalize()
        result = transform(x)

        # Check each channel has zero mean and unit variance
        for c in range(2):
            assert abs(result[c].mean()) < 0.01
            assert abs(result[c].std() - 1.0) < 0.01

    def test_tensor_input(self):
        x = torch.randn(2, 128) * 5 + 3
        transform = Normalize()
        result = transform(x)

        assert isinstance(result, torch.Tensor)
        for c in range(2):
            assert abs(result[c].mean()) < 0.01
            assert abs(result[c].std() - 1.0) < 0.01


class TestComplexConversion:
    def test_complex_to_iq(self):
        complex_signal = np.random.randn(128) + 1j * np.random.randn(128)
        transform = ComplexToIQ()
        result = transform(complex_signal)

        assert result.shape == (2, 128)
        np.testing.assert_allclose(result[0], complex_signal.real)
        np.testing.assert_allclose(result[1], complex_signal.imag)

    def test_iq_to_complex(self):
        iq = np.random.randn(2, 128).astype(np.float32)
        transform = IQToComplex()
        result = transform(iq)

        assert np.iscomplexobj(result)
        assert result.shape == (128,)
        np.testing.assert_allclose(result.real, iq[0])
        np.testing.assert_allclose(result.imag, iq[1])

    def test_roundtrip(self):
        original = np.random.randn(128) + 1j * np.random.randn(128)

        to_iq = ComplexToIQ()
        to_complex = IQToComplex()

        iq = to_iq(original)
        recovered = to_complex(iq)

        np.testing.assert_allclose(recovered, original, rtol=1e-5)


class TestCompose:
    def test_chain_transforms(self):
        x = np.random.randn(2, 128).astype(np.float32) * 5

        transform = Compose([PowerNormalize(), ToTensor()])
        result = transform(x)

        assert isinstance(result, torch.Tensor)
        power = torch.mean(result[0] ** 2 + result[1] ** 2)
        assert abs(power - 1.0) < 0.01

    def test_repr(self):
        transform = Compose([PowerNormalize(), ToTensor()])
        repr_str = repr(transform)

        assert "Compose" in repr_str
        assert "PowerNormalize" in repr_str
        assert "ToTensor" in repr_str
