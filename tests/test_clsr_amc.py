"""Tests for CLSR-AMC model."""

import numpy as np
import pytest
import torch

from robust_amc.models import CLSRAMC, CLSRAMCLoss, create_clsr_amc
from robust_amc.models.clsr_amc import CLSRAMCEncoder


class TestCLSRAMCEncoder:
    def test_output_shape(self):
        """Encoder output should have correct shape."""
        encoder = CLSRAMCEncoder(n_filters=4, n_stages=5)
        x = torch.randn(32, 2, 128)

        out = encoder(x)

        assert out.shape[0] == 32
        assert out.shape[1] == encoder.output_dim

    def test_feature_extraction(self):
        """Feature extraction should produce amplitude and phase."""
        encoder = CLSRAMCEncoder()
        x = torch.randn(16, 2, 128)

        amp, phase = encoder.extract_features(x)

        assert amp.shape == (16, 1, 128)
        assert phase.shape == (16, 1, 128)
        # Amplitude should be non-negative
        assert (amp >= 0).all()
        # Phase should be in [-pi, pi]
        assert (phase >= -np.pi).all() and (phase <= np.pi).all()

    def test_gradient_flow(self):
        """Gradients should flow through encoder."""
        encoder = CLSRAMCEncoder()
        x = torch.randn(16, 2, 128, requires_grad=True)

        out = encoder(x)
        out.sum().backward()

        assert x.grad is not None


class TestCLSRAMC:
    def test_forward_classification(self):
        """Forward pass should produce correct logit shape."""
        model = CLSRAMC(num_classes=11)
        x = torch.randn(32, 2, 128)

        logits = model(x)

        assert logits.shape == (32, 11)

    def test_forward_all_outputs(self):
        """Forward with return_all should return all components."""
        model = CLSRAMC(num_classes=11)
        x = torch.randn(16, 2, 128)

        outputs = model(x, return_all=True)

        assert "embeddings" in outputs
        assert "projections" in outputs
        assert "reconstruction" in outputs
        assert "logits" in outputs

        assert outputs["embeddings"].shape[0] == 16
        assert outputs["projections"].shape[0] == 16
        assert outputs["reconstruction"].shape == (16, 2, 128)
        assert outputs["logits"].shape == (16, 11)

    def test_encode(self):
        """Encode should produce embeddings."""
        model = CLSRAMC(num_classes=11)
        x = torch.randn(32, 2, 128)

        z = model.encode(x)

        assert z.shape[0] == 32
        assert z.shape[1] == model.embedding_dim

    def test_project(self):
        """Project should produce projections for contrastive loss."""
        model = CLSRAMC(num_classes=11, projection_dim=64)
        z = torch.randn(32, model.embedding_dim)

        p = model.project(z)

        assert p.shape == (32, 64)

    def test_decode(self):
        """Decode should reconstruct signal shape."""
        model = CLSRAMC(num_classes=11)
        z = torch.randn(32, model.embedding_dim)

        x_recon = model.decode(z)

        assert x_recon.shape == (32, 2, 128)

    def test_classify(self):
        """Classify should produce logits."""
        model = CLSRAMC(num_classes=11)
        z = torch.randn(32, model.embedding_dim)

        logits = model.classify(z)

        assert logits.shape == (32, 11)

    def test_forward_contrastive(self):
        """Forward contrastive should return embeddings and projections."""
        model = CLSRAMC(num_classes=11)
        x_i = torch.randn(16, 2, 128)
        x_j = torch.randn(16, 2, 128)

        z_i, z_j, p_i, p_j = model.forward_contrastive(x_i, x_j)

        assert z_i.shape == z_j.shape
        assert p_i.shape == p_j.shape
        assert z_i.shape[0] == 16
        assert p_i.shape[0] == 16

    def test_get_embeddings(self):
        """Get embeddings should work like encode."""
        model = CLSRAMC(num_classes=11)
        x = torch.randn(32, 2, 128)

        z1 = model.encode(x)
        z2 = model.get_embeddings(x)

        torch.testing.assert_close(z1, z2)

    def test_gradient_flow(self):
        """Gradients should flow through entire model."""
        model = CLSRAMC(num_classes=11)
        x = torch.randn(16, 2, 128, requires_grad=True)

        outputs = model(x, return_all=True)
        loss = outputs["logits"].sum() + outputs["reconstruction"].sum()
        loss.backward()

        assert x.grad is not None

    def test_decoder_types(self):
        """Both decoder types should work."""
        x = torch.randn(16, 2, 128)

        model_linear = CLSRAMC(num_classes=11, decoder_type="linear")
        model_conv = CLSRAMC(num_classes=11, decoder_type="conv")

        out_linear = model_linear(x, return_all=True)
        out_conv = model_conv(x, return_all=True)

        assert out_linear["reconstruction"].shape == (16, 2, 128)
        assert out_conv["reconstruction"].shape == (16, 2, 128)


class TestCLSRAMCLoss:
    def test_all_components(self):
        """Loss should return all components."""
        loss_fn = CLSRAMCLoss(
            contrastive_weight=1.0,
            reconstruction_weight=1.0,
            classification_weight=1.0,
        )

        p_i = torch.randn(16, 64)
        p_j = torch.randn(16, 64)
        x_recon = torch.randn(16, 2, 128)
        x_orig = torch.randn(16, 2, 128)
        logits = torch.randn(16, 11)
        labels = torch.randint(0, 11, (16,))

        losses = loss_fn(p_i, p_j, x_recon, x_orig, logits, labels)

        assert "total" in losses
        assert "contrastive" in losses
        assert "reconstruction" in losses
        assert "classification" in losses

        assert torch.isfinite(losses["total"])
        assert losses["total"].item() > 0

    def test_zero_weights(self):
        """Zero weights should disable components."""
        loss_contrastive = CLSRAMCLoss(
            contrastive_weight=1.0, reconstruction_weight=0.0, classification_weight=0.0
        )
        loss_reconstruction = CLSRAMCLoss(
            contrastive_weight=0.0, reconstruction_weight=1.0, classification_weight=0.0
        )
        loss_classification = CLSRAMCLoss(
            contrastive_weight=0.0, reconstruction_weight=0.0, classification_weight=1.0
        )

        p_i = torch.randn(16, 64)
        p_j = torch.randn(16, 64)
        x_recon = torch.randn(16, 2, 128)
        x_orig = torch.randn(16, 2, 128)
        logits = torch.randn(16, 11)
        labels = torch.randint(0, 11, (16,))

        losses_con = loss_contrastive(p_i, p_j, x_recon, x_orig, logits, labels)
        losses_rec = loss_reconstruction(p_i, p_j, x_recon, x_orig, logits, labels)
        losses_cls = loss_classification(p_i, p_j, x_recon, x_orig, logits, labels)

        # Total should equal the non-zero component
        assert torch.isclose(losses_con["total"], losses_con["contrastive"])
        assert torch.isclose(losses_rec["total"], losses_rec["reconstruction"])
        assert torch.isclose(losses_cls["total"], losses_cls["classification"])

    def test_gradient_flow(self):
        """Gradients should flow through loss."""
        loss_fn = CLSRAMCLoss()

        p_i = torch.randn(16, 64, requires_grad=True)
        p_j = torch.randn(16, 64, requires_grad=True)
        x_recon = torch.randn(16, 2, 128, requires_grad=True)
        x_orig = torch.randn(16, 2, 128)
        logits = torch.randn(16, 11, requires_grad=True)
        labels = torch.randint(0, 11, (16,))

        losses = loss_fn(p_i, p_j, x_recon, x_orig, logits, labels)
        losses["total"].backward()

        assert p_i.grad is not None
        assert x_recon.grad is not None
        assert logits.grad is not None


class TestCreateCLSRAMC:
    def test_default_variant(self):
        """Default variant should create valid model."""
        model = create_clsr_amc(num_classes=11, variant="default")

        x = torch.randn(16, 2, 128)
        out = model(x)

        assert out.shape == (16, 11)

    def test_small_variant(self):
        """Small variant should create smaller model."""
        model_small = create_clsr_amc(num_classes=11, variant="small")
        model_default = create_clsr_amc(num_classes=11, variant="default")

        params_small = sum(p.numel() for p in model_small.parameters())
        params_default = sum(p.numel() for p in model_default.parameters())

        assert params_small < params_default

    def test_large_variant(self):
        """Large variant should create larger model."""
        model_large = create_clsr_amc(num_classes=11, variant="large")
        model_default = create_clsr_amc(num_classes=11, variant="default")

        params_large = sum(p.numel() for p in model_large.parameters())
        params_default = sum(p.numel() for p in model_default.parameters())

        assert params_large > params_default

    def test_invalid_variant(self):
        """Invalid variant should raise error."""
        with pytest.raises(ValueError):
            create_clsr_amc(variant="invalid")

    def test_custom_num_classes(self):
        """Should work with different number of classes."""
        for n_classes in [5, 11, 20]:
            model = create_clsr_amc(num_classes=n_classes)
            x = torch.randn(8, 2, 128)
            out = model(x)
            assert out.shape == (8, n_classes)


class TestIntegration:
    """Integration tests for full training pipeline."""

    def test_training_step(self):
        """Full training step should work."""
        model = create_clsr_amc(num_classes=11)
        loss_fn = CLSRAMCLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Simulate batch
        x_i = torch.randn(16, 2, 128)
        x_j = torch.randn(16, 2, 128)
        x_orig = torch.randn(16, 2, 128)
        labels = torch.randint(0, 11, (16,))

        # Forward
        z_i, z_j, p_i, p_j = model.forward_contrastive(x_i, x_j)
        x_recon = model.decode(z_i)
        logits = model.classify(z_i)

        # Loss
        losses = loss_fn(p_i, p_j, x_recon, x_orig, logits, labels)

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        # Should complete without error
        assert True

    def test_eval_mode(self):
        """Model should work in eval mode."""
        model = create_clsr_amc(num_classes=11)
        model.eval()

        x = torch.randn(16, 2, 128)

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (16, 11)