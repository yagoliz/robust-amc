"""Tests for loss functions."""

import numpy as np
import pytest
import torch

from robust_amc.losses import (
    NTXentLoss,
    SupConLoss,
    ProjectionHead,
    ReconstructionLoss,
    SignalDecoder,
    LightweightDecoder,
)


class TestNTXentLoss:
    def test_output_is_scalar(self):
        """Loss should be a scalar."""
        loss_fn = NTXentLoss(temperature=0.5)
        z_i = torch.randn(32, 128)
        z_j = torch.randn(32, 128)

        loss = loss_fn(z_i, z_j)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_identical_pairs_low_loss(self):
        """Identical pairs should have lower loss than random."""
        loss_fn = NTXentLoss(temperature=0.5)

        # Identical views
        z = torch.randn(32, 128)
        loss_identical = loss_fn(z, z)

        # Random views
        z_i = torch.randn(32, 128)
        z_j = torch.randn(32, 128)
        loss_random = loss_fn(z_i, z_j)

        assert loss_identical < loss_random

    def test_temperature_effect(self):
        """Lower temperature should make loss more peaked."""
        z_i = torch.randn(32, 128)
        z_j = z_i + 0.1 * torch.randn(32, 128)  # Similar but not identical

        loss_low_temp = NTXentLoss(temperature=0.1)(z_i, z_j)
        loss_high_temp = NTXentLoss(temperature=1.0)(z_i, z_j)

        # Lower temperature should give lower loss for similar pairs
        assert loss_low_temp < loss_high_temp

    def test_normalization_option(self):
        """Normalization option should work."""
        z_i = torch.randn(32, 128) * 10  # Large magnitude
        z_j = torch.randn(32, 128) * 10

        loss_normalized = NTXentLoss(normalize=True)(z_i, z_j)
        loss_unnormalized = NTXentLoss(normalize=False)(z_i, z_j)

        # Both should compute without error
        assert torch.isfinite(loss_normalized)
        assert torch.isfinite(loss_unnormalized)

    def test_gradient_flow(self):
        """Gradients should flow through the loss."""
        loss_fn = NTXentLoss()
        z_i = torch.randn(16, 64, requires_grad=True)
        z_j = torch.randn(16, 64, requires_grad=True)

        loss = loss_fn(z_i, z_j)
        loss.backward()

        assert z_i.grad is not None
        assert z_j.grad is not None

    def test_batch_size_one(self):
        """Should handle batch size of 1."""
        loss_fn = NTXentLoss()
        z_i = torch.randn(1, 64)
        z_j = torch.randn(1, 64)

        # With batch size 1, there are no negatives
        # This is a degenerate case but shouldn't crash
        loss = loss_fn(z_i, z_j)
        assert torch.isfinite(loss)


class TestSupConLoss:
    def test_output_is_scalar(self):
        """Loss should be a scalar."""
        loss_fn = SupConLoss(temperature=0.5)
        z_i = torch.randn(32, 128)
        z_j = torch.randn(32, 128)
        labels = torch.randint(0, 5, (32,))

        loss = loss_fn(z_i, z_j, labels)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_labels_affect_loss(self):
        """Different label configurations should produce different losses."""
        loss_fn = SupConLoss(temperature=0.5)

        z_i = torch.randn(16, 64)
        z_j = z_i + 0.1 * torch.randn(16, 64)

        # All same label - many positives per sample
        same_labels = torch.zeros(16, dtype=torch.long)
        loss_same = loss_fn(z_i, z_j, same_labels)

        # All different labels - only augmented pair is positive
        diff_labels = torch.arange(16)
        loss_diff = loss_fn(z_i, z_j, diff_labels)

        # Both should be finite and positive
        assert torch.isfinite(loss_same) and loss_same.item() >= 0
        assert torch.isfinite(loss_diff) and loss_diff.item() >= 0

        # Losses should be different (labels matter)
        assert not torch.isclose(loss_same, loss_diff, atol=0.01)

    def test_gradient_flow(self):
        """Gradients should flow through the loss."""
        loss_fn = SupConLoss()
        z_i = torch.randn(16, 64, requires_grad=True)
        z_j = torch.randn(16, 64, requires_grad=True)
        labels = torch.randint(0, 3, (16,))

        loss = loss_fn(z_i, z_j, labels)
        loss.backward()

        assert z_i.grad is not None
        assert z_j.grad is not None


class TestProjectionHead:
    def test_output_shape(self):
        """Output should have correct shape."""
        proj = ProjectionHead(input_dim=128, hidden_dim=256, output_dim=64)
        x = torch.randn(32, 128)

        out = proj(x)

        assert out.shape == (32, 64)

    def test_different_num_layers(self):
        """Should work with different numbers of layers."""
        x = torch.randn(16, 64)

        for num_layers in [2, 3, 4]:
            proj = ProjectionHead(
                input_dim=64, hidden_dim=128, output_dim=32, num_layers=num_layers
            )
            out = proj(x)
            assert out.shape == (16, 32)

    def test_gradient_flow(self):
        """Gradients should flow through projection head."""
        proj = ProjectionHead(input_dim=64, hidden_dim=128, output_dim=32)
        x = torch.randn(16, 64, requires_grad=True)

        out = proj(x)
        out.sum().backward()

        assert x.grad is not None


class TestReconstructionLoss:
    def test_mse_loss(self):
        """MSE loss should work correctly."""
        loss_fn = ReconstructionLoss(mse_weight=1.0)
        x_recon = torch.randn(32, 2, 128)
        x_orig = torch.randn(32, 2, 128)

        loss = loss_fn(x_recon, x_orig)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_identical_signals_zero_loss(self):
        """Identical signals should have near-zero loss."""
        loss_fn = ReconstructionLoss(mse_weight=1.0)
        x = torch.randn(32, 2, 128)

        loss = loss_fn(x, x)

        assert loss.item() < 1e-6

    def test_amplitude_loss(self):
        """Amplitude loss component should work."""
        loss_fn = ReconstructionLoss(
            mse_weight=0.0, amplitude_weight=1.0, phase_weight=0.0
        )
        x_recon = torch.randn(16, 2, 64)
        x_orig = torch.randn(16, 2, 64)

        loss = loss_fn(x_recon, x_orig)
        assert torch.isfinite(loss)

    def test_phase_loss(self):
        """Phase loss component should work."""
        loss_fn = ReconstructionLoss(
            mse_weight=0.0, amplitude_weight=0.0, phase_weight=1.0
        )
        x_recon = torch.randn(16, 2, 64)
        x_orig = torch.randn(16, 2, 64)

        loss = loss_fn(x_recon, x_orig)
        assert torch.isfinite(loss)

    def test_combined_losses(self):
        """All loss components should combine correctly."""
        loss_fn = ReconstructionLoss(
            mse_weight=1.0, amplitude_weight=0.5, phase_weight=0.5
        )
        x_recon = torch.randn(16, 2, 64)
        x_orig = torch.randn(16, 2, 64)

        loss = loss_fn(x_recon, x_orig)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Gradients should flow through reconstruction loss."""
        loss_fn = ReconstructionLoss()
        x_recon = torch.randn(16, 2, 64, requires_grad=True)
        x_orig = torch.randn(16, 2, 64)

        loss = loss_fn(x_recon, x_orig)
        loss.backward()

        assert x_recon.grad is not None


class TestSignalDecoder:
    def test_output_shape(self):
        """Output should have correct shape."""
        decoder = SignalDecoder(embedding_dim=128, output_len=128)
        z = torch.randn(32, 128)

        out = decoder(z)

        assert out.shape == (32, 2, 128)

    def test_different_output_lengths(self):
        """Should work with different output lengths."""
        for out_len in [64, 128, 256]:
            decoder = SignalDecoder(embedding_dim=64, output_len=out_len)
            z = torch.randn(16, 64)
            out = decoder(z)
            assert out.shape == (16, 2, out_len)

    def test_gradient_flow(self):
        """Gradients should flow through decoder."""
        decoder = SignalDecoder(embedding_dim=64, output_len=64)
        z = torch.randn(16, 64, requires_grad=True)

        out = decoder(z)
        out.sum().backward()

        assert z.grad is not None


class TestLightweightDecoder:
    def test_output_shape(self):
        """Output should have correct shape."""
        decoder = LightweightDecoder(embedding_dim=128, output_len=128)
        z = torch.randn(32, 128)

        out = decoder(z)

        assert out.shape == (32, 2, 128)

    def test_different_output_lengths(self):
        """Should work with different output lengths."""
        for out_len in [64, 128, 256]:
            decoder = LightweightDecoder(embedding_dim=64, output_len=out_len)
            z = torch.randn(16, 64)
            out = decoder(z)
            assert out.shape == (16, 2, out_len)

    def test_gradient_flow(self):
        """Gradients should flow through decoder."""
        decoder = LightweightDecoder(embedding_dim=64)
        z = torch.randn(16, 64, requires_grad=True)

        out = decoder(z)
        out.sum().backward()

        assert z.grad is not None