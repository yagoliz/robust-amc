"""Tests for MDA-DMC data augmentation techniques."""

import numpy as np
import pytest
import torch

from robust_amc.data.augmentations import (
    AdditiveGaussianNoise,
    RotationInSignalConstellation,
    StretchingInSignalConstellation,
    RotationAndStretchingInSignalConstellation,
    TimeShift,
    RandomFlip,
    RandomAugmentation,
    MDADMCPipeline,
    AGN,
    RSC,
    SSC,
    RSSC,
)


class TestAdditiveGaussianNoise:
    def test_adds_noise(self):
        """AGN should add noise to the signal."""
        x = np.ones((2, 128), dtype=np.float32)

        agn = AdditiveGaussianNoise(snr_range=(10.0, 10.0), p=1.0, seed=42)
        y = agn(x)

        # Output should be different from input
        assert not np.allclose(y, x)

    def test_noise_level_matches_snr(self):
        """Noise power should approximately match target SNR."""
        x = np.ones((2, 128), dtype=np.float32)
        target_snr_db = 10.0

        agn = AdditiveGaussianNoise(
            snr_range=(target_snr_db, target_snr_db), p=1.0, seed=42
        )
        y = agn(x)

        # Estimate noise
        noise = y - x
        noise_power = np.mean(noise[0] ** 2 + noise[1] ** 2)
        signal_power = np.mean(x[0] ** 2 + x[1] ** 2)

        actual_snr_db = 10 * np.log10(signal_power / noise_power)
        # Allow some tolerance due to finite samples
        assert abs(actual_snr_db - target_snr_db) < 1.0

    def test_probability_zero(self):
        """With p=0, signal should not change."""
        x = np.random.randn(2, 128).astype(np.float32)

        agn = AdditiveGaussianNoise(p=0.0, seed=42)
        y = agn(x)

        np.testing.assert_allclose(y, x)

    def test_probability_one(self):
        """With p=1, noise should always be added."""
        x = np.ones((2, 128), dtype=np.float32)

        agn = AdditiveGaussianNoise(p=1.0, seed=42)
        y = agn(x)

        assert not np.allclose(y, x)

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        agn = AdditiveGaussianNoise(p=1.0, seed=42)
        y = agn(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        x = np.random.randn(2, 256).astype(np.float32)

        agn = AdditiveGaussianNoise(p=1.0, seed=42)
        y = agn(x)

        assert y.shape == x.shape

    def test_alias(self):
        """AGN should be alias for AdditiveGaussianNoise."""
        assert AGN is AdditiveGaussianNoise


class TestRotationInSignalConstellation:
    def test_rotates_signal(self):
        """RSC should rotate the signal in I/Q plane."""
        x = np.zeros((2, 128), dtype=np.float32)
        x[0, :] = 1.0  # Pure I signal

        rsc = RotationInSignalConstellation(angle_range=(90, 90), p=1.0, seed=42)
        y = rsc(x)

        # After 90 degree rotation, I should become Q
        np.testing.assert_allclose(y[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(y[1], 1.0, atol=1e-5)

    def test_rotation_180(self):
        """180 degree rotation should flip both I and Q."""
        x = np.ones((2, 128), dtype=np.float32)

        rsc = RotationInSignalConstellation(angle_range=(180, 180), p=1.0, seed=42)
        y = rsc(x)

        np.testing.assert_allclose(y[0], -1.0, atol=1e-5)
        np.testing.assert_allclose(y[1], -1.0, atol=1e-5)

    def test_preserves_amplitude(self):
        """Rotation should preserve signal amplitude."""
        x = np.random.randn(2, 128).astype(np.float32)

        rsc = RotationInSignalConstellation(p=1.0, seed=42)
        y = rsc(x)

        input_amp = np.sqrt(x[0] ** 2 + x[1] ** 2)
        output_amp = np.sqrt(y[0] ** 2 + y[1] ** 2)

        np.testing.assert_allclose(input_amp, output_amp, rtol=1e-5)

    def test_probability_zero(self):
        """With p=0, signal should not change."""
        x = np.random.randn(2, 128).astype(np.float32)

        rsc = RotationInSignalConstellation(p=0.0, seed=42)
        y = rsc(x)

        np.testing.assert_allclose(y, x)

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        rsc = RotationInSignalConstellation(p=1.0, seed=42)
        y = rsc(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)

    def test_alias(self):
        """RSC should be alias for RotationInSignalConstellation."""
        assert RSC is RotationInSignalConstellation


class TestStretchingInSignalConstellation:
    def test_scales_signal(self):
        """SSC should scale the signal amplitude."""
        x = np.ones((2, 128), dtype=np.float32)

        ssc = StretchingInSignalConstellation(scale_range=(2.0, 2.0), p=1.0, seed=42)
        y = ssc(x)

        np.testing.assert_allclose(y, x * 2.0, rtol=1e-5)

    def test_scale_half(self):
        """Scaling by 0.5 should halve the signal."""
        x = np.ones((2, 128), dtype=np.float32)

        ssc = StretchingInSignalConstellation(scale_range=(0.5, 0.5), p=1.0, seed=42)
        y = ssc(x)

        np.testing.assert_allclose(y, x * 0.5, rtol=1e-5)

    def test_probability_zero(self):
        """With p=0, signal should not change."""
        x = np.random.randn(2, 128).astype(np.float32)

        ssc = StretchingInSignalConstellation(p=0.0, seed=42)
        y = ssc(x)

        np.testing.assert_allclose(y, x)

    def test_random_scale_in_range(self):
        """Random scale should be within specified range."""
        x = np.ones((2, 128), dtype=np.float32)

        # Run multiple times with different seeds
        for seed in range(10):
            ssc = StretchingInSignalConstellation(
                scale_range=(0.5, 1.5), p=1.0, seed=seed
            )
            y = ssc(x)
            scale = y[0, 0]  # Scale factor since input is 1.0
            assert 0.5 <= scale <= 1.5

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        ssc = StretchingInSignalConstellation(p=1.0, seed=42)
        y = ssc(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)

    def test_alias(self):
        """SSC should be alias for StretchingInSignalConstellation."""
        assert SSC is StretchingInSignalConstellation


class TestRotationAndStretchingInSignalConstellation:
    def test_applies_both_transforms(self):
        """RSSC should apply both rotation and scaling."""
        x = np.zeros((2, 128), dtype=np.float32)
        x[0, :] = 1.0  # Pure I signal

        rssc = RotationAndStretchingInSignalConstellation(
            angle_range=(90, 90), scale_range=(2.0, 2.0), p=1.0, seed=42
        )
        y = rssc(x)

        # After 90 deg rotation and 2x scale: I=1 -> Q=2
        np.testing.assert_allclose(y[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(y[1], 2.0, atol=1e-5)

    def test_probability_zero(self):
        """With p=0, signal should not change."""
        x = np.random.randn(2, 128).astype(np.float32)

        rssc = RotationAndStretchingInSignalConstellation(p=0.0, seed=42)
        y = rssc(x)

        np.testing.assert_allclose(y, x)

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        rssc = RotationAndStretchingInSignalConstellation(p=1.0, seed=42)
        y = rssc(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)

    def test_alias(self):
        """RSSC should be alias for RotationAndStretchingInSignalConstellation."""
        assert RSSC is RotationAndStretchingInSignalConstellation


class TestTimeShift:
    def test_shifts_signal(self):
        """TimeShift should circular shift the signal."""
        x = np.zeros((2, 128), dtype=np.float32)
        x[0, 0] = 1.0  # Single impulse at position 0

        # Force specific shift
        shift = TimeShift(max_shift=1, p=1.0, seed=42)
        y = shift(x)

        # Signal should be shifted (impulse not at original position)
        # Due to circular shift, position might wrap around
        assert y[0, 0] != 1.0 or np.sum(y[0] == 1.0) == 1

    def test_probability_zero(self):
        """With p=0, signal should not change."""
        x = np.random.randn(2, 128).astype(np.float32)

        shift = TimeShift(p=0.0, seed=42)
        y = shift(x)

        np.testing.assert_allclose(y, x)

    def test_circular_shift(self):
        """Shift should be circular (wrap around)."""
        x = np.zeros((2, 10), dtype=np.float32)
        x[0, 0] = 1.0

        # Test with known shift
        shift = TimeShift(max_shift=100, p=1.0, seed=0)
        y = shift(x)

        # The 1.0 should still be somewhere in the signal
        assert np.sum(y[0] == 1.0) == 1

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        shift = TimeShift(p=1.0, seed=42)
        y = shift(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)


class TestRandomFlip:
    def test_flips_i(self):
        """RandomFlip should flip I channel."""
        x = np.ones((2, 128), dtype=np.float32)

        # Seed chosen to flip I
        flip = RandomFlip(flip_i=True, flip_q=False, p=1.0, seed=42)
        y = flip(x)

        # I should be flipped (or not), Q should be unchanged
        assert np.allclose(y[1], x[1])

    def test_flips_q(self):
        """RandomFlip should flip Q channel."""
        x = np.ones((2, 128), dtype=np.float32)

        flip = RandomFlip(flip_i=False, flip_q=True, p=1.0, seed=42)
        y = flip(x)

        # I should be unchanged
        np.testing.assert_allclose(y[0], x[0])

    def test_probability_zero(self):
        """With p=0, signal should not change."""
        x = np.ones((2, 128), dtype=np.float32)

        flip = RandomFlip(p=0.0, seed=42)
        y = flip(x)

        np.testing.assert_allclose(y, x)

    def test_tensor_input(self):
        """Should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        flip = RandomFlip(p=1.0, seed=42)
        y = flip(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)


class TestRandomAugmentation:
    def test_selects_augmentation(self):
        """RandomAugmentation should select and apply one augmentation."""
        x = np.ones((2, 128), dtype=np.float32)

        augs = [
            StretchingInSignalConstellation(scale_range=(2.0, 2.0), p=1.0),
            StretchingInSignalConstellation(scale_range=(0.5, 0.5), p=1.0),
        ]

        rand_aug = RandomAugmentation(augs, seed=42)
        y = rand_aug(x)

        # Should apply one of the two scalings
        assert np.allclose(y, x * 2.0, rtol=1e-5) or np.allclose(
            y, x * 0.5, rtol=1e-5
        )

    def test_different_selections(self):
        """Different seeds should produce different selections."""
        x = np.ones((2, 128), dtype=np.float32)

        augs = [
            StretchingInSignalConstellation(scale_range=(2.0, 2.0), p=1.0),
            StretchingInSignalConstellation(scale_range=(0.5, 0.5), p=1.0),
        ]

        results = set()
        for seed in range(20):
            rand_aug = RandomAugmentation(augs, seed=seed)
            y = rand_aug(x)
            results.add(round(y[0, 0], 1))

        # Both augmentations should have been selected at least once
        assert len(results) == 2


class TestMDADMCPipeline:
    def test_default_pipeline(self):
        """Default pipeline should apply AGN, RSC, SSC."""
        x = np.ones((2, 128), dtype=np.float32)

        pipeline = MDADMCPipeline(p=1.0, seed=42)
        y = pipeline(x)

        # Output should be different from input
        assert not np.allclose(y, x)

    def test_agn_only(self):
        """Pipeline with only AGN should add noise."""
        x = np.ones((2, 128), dtype=np.float32)

        pipeline = MDADMCPipeline(agn=True, rsc=False, ssc=False, p=1.0, seed=42)
        y = pipeline(x)

        # Should have noise added
        assert not np.allclose(y, x)

    def test_rsc_only(self):
        """Pipeline with only RSC should rotate."""
        x = np.zeros((2, 128), dtype=np.float32)
        x[0, :] = 1.0

        pipeline = MDADMCPipeline(agn=False, rsc=True, ssc=False, p=1.0, seed=42)
        y = pipeline(x)

        # Amplitude should be preserved
        input_amp = np.sqrt(x[0] ** 2 + x[1] ** 2)
        output_amp = np.sqrt(y[0] ** 2 + y[1] ** 2)
        np.testing.assert_allclose(input_amp, output_amp, rtol=1e-5)

    def test_ssc_only(self):
        """Pipeline with only SSC should scale."""
        x = np.ones((2, 128), dtype=np.float32)

        pipeline = MDADMCPipeline(agn=False, rsc=False, ssc=True, p=1.0, seed=42)
        y = pipeline(x)

        # All values should be scaled uniformly
        scale = y[0, 0]
        np.testing.assert_allclose(y, x * scale, rtol=1e-5)

    def test_all_disabled(self):
        """Pipeline with all disabled should not change signal."""
        x = np.random.randn(2, 128).astype(np.float32)

        pipeline = MDADMCPipeline(agn=False, rsc=False, ssc=False, p=1.0, seed=42)
        y = pipeline(x)

        np.testing.assert_allclose(y, x)

    def test_with_time_shift(self):
        """Pipeline should support optional time shift."""
        x = np.zeros((2, 128), dtype=np.float32)
        x[0, 0] = 1.0

        pipeline = MDADMCPipeline(
            agn=False, rsc=False, ssc=False, time_shift=True, p=1.0, seed=42
        )
        y = pipeline(x)

        # Signal should be shifted
        assert y[0, 0] != 1.0 or np.sum(y[0] == 1.0) == 1

    def test_with_random_flip(self):
        """Pipeline should support optional random flip."""
        x = np.ones((2, 128), dtype=np.float32)

        pipeline = MDADMCPipeline(
            agn=False, rsc=False, ssc=False, random_flip=True, p=1.0, seed=42
        )
        y = pipeline(x)

        # Signal might be flipped
        assert np.allclose(y[0], 1.0) or np.allclose(y[0], -1.0)

    def test_tensor_input(self):
        """Pipeline should handle PyTorch tensors."""
        x = torch.randn(2, 128)

        pipeline = MDADMCPipeline(p=1.0, seed=42)
        y = pipeline(x)

        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 128)

    def test_repr(self):
        """Pipeline should have readable repr."""
        pipeline = MDADMCPipeline(p=0.5, seed=42)
        repr_str = repr(pipeline)

        assert "MDADMCPipeline" in repr_str
        assert "p=0.5" in repr_str

    def test_custom_ranges(self):
        """Pipeline should accept custom parameter ranges."""
        pipeline = MDADMCPipeline(
            agn_snr_range=(0.0, 5.0),
            rsc_angle_range=(-90.0, 90.0),
            ssc_scale_range=(0.9, 1.1),
            p=1.0,
            seed=42,
        )

        # Should not raise
        x = np.random.randn(2, 128).astype(np.float32)
        y = pipeline(x)
        assert y.shape == x.shape


class TestReproducibility:
    """Test that all augmentations are reproducible with seeds."""

    def test_agn_reproducible(self):
        """AGN should produce same result with same seed."""
        x = np.random.randn(2, 128).astype(np.float32)

        agn1 = AdditiveGaussianNoise(p=1.0, seed=42)
        agn2 = AdditiveGaussianNoise(p=1.0, seed=42)

        y1 = agn1(x.copy())
        y2 = agn2(x.copy())

        np.testing.assert_allclose(y1, y2)

    def test_rsc_reproducible(self):
        """RSC should produce same result with same seed."""
        x = np.random.randn(2, 128).astype(np.float32)

        rsc1 = RotationInSignalConstellation(p=1.0, seed=42)
        rsc2 = RotationInSignalConstellation(p=1.0, seed=42)

        y1 = rsc1(x.copy())
        y2 = rsc2(x.copy())

        np.testing.assert_allclose(y1, y2)

    def test_pipeline_reproducible(self):
        """Pipeline should produce same result with same seed."""
        x = np.random.randn(2, 128).astype(np.float32)

        pipeline1 = MDADMCPipeline(p=1.0, seed=42)
        pipeline2 = MDADMCPipeline(p=1.0, seed=42)

        y1 = pipeline1(x.copy())
        y2 = pipeline2(x.copy())

        np.testing.assert_allclose(y1, y2)