"""Tests for model architectures."""

import pytest
import torch

from robust_amc.models.pf_cnn import PFCNN, FeatureBranch, create_pfcnn


class TestFeatureBranch:
    def test_output_shape(self):
        branch = FeatureBranch(in_channels=1, n_filters=4, n_stages=5)
        x = torch.randn(8, 1, 128)
        out = branch(x)

        assert out.shape == (8, branch.output_dim)

    def test_output_dim_calculation(self):
        branch = FeatureBranch(in_channels=1, n_filters=4, n_stages=5)
        # Output dim should be n_filters * 2^(n_stages-1) = 4 * 16 = 64
        assert branch.output_dim == 64

    def test_different_configs(self):
        configs = [
            {"n_filters": 2, "n_stages": 3},
            {"n_filters": 8, "n_stages": 4},
            {"n_filters": 4, "n_stages": 6},
        ]

        for cfg in configs:
            branch = FeatureBranch(**cfg)
            x = torch.randn(4, 1, 128)
            out = branch(x)

            expected_dim = cfg["n_filters"] * (2 ** (cfg["n_stages"] - 1))
            assert out.shape == (4, expected_dim)


class TestPFCNN:
    def test_forward_shape(self):
        model = PFCNN(num_classes=11)
        x = torch.randn(8, 2, 128)
        out = model(x)

        assert out.shape == (8, 11)

    def test_extract_features(self):
        model = PFCNN(num_classes=11)
        x = torch.randn(8, 2, 128)
        amp, phase = model.extract_features(x)

        assert amp.shape == (8, 1, 128)
        assert phase.shape == (8, 1, 128)

        # Amplitude should be non-negative
        assert (amp >= 0).all()

        # Phase should be in [-pi, pi]
        assert (phase >= -torch.pi).all()
        assert (phase <= torch.pi).all()

    def test_get_embeddings(self):
        model = PFCNN(num_classes=11, n_filters=4, n_stages=5)
        x = torch.randn(8, 2, 128)
        emb = model.get_embeddings(x)

        # Embedding dim should be 2 * (n_filters * 2^(n_stages-1)) = 2 * 64 = 128
        assert emb.shape == (8, 128)

    def test_gradient_flow(self):
        model = PFCNN(num_classes=11)
        x = torch.randn(8, 2, 128, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check gradients flow to input
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_batch_sizes(self):
        model = PFCNN(num_classes=11)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 2, 128)
            out = model(x)
            assert out.shape == (batch_size, 11)


class TestCreatePFCNN:
    def test_default_variant(self):
        model = create_pfcnn(num_classes=11, variant="default")
        assert isinstance(model, PFCNN)
        assert model.num_classes == 11

    def test_small_variant(self):
        model = create_pfcnn(num_classes=11, variant="small")
        x = torch.randn(4, 2, 128)
        out = model(x)
        assert out.shape == (4, 11)

    def test_large_variant(self):
        model = create_pfcnn(num_classes=11, variant="large")
        x = torch.randn(4, 2, 128)
        out = model(x)
        assert out.shape == (4, 11)

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            create_pfcnn(variant="invalid")

    def test_custom_num_classes(self):
        for n_classes in [5, 11, 24]:
            model = create_pfcnn(num_classes=n_classes)
            x = torch.randn(4, 2, 128)
            out = model(x)
            assert out.shape == (4, n_classes)


class TestModelProperties:
    def test_parameter_count_reasonable(self):
        model = create_pfcnn(num_classes=11, variant="default")
        n_params = sum(p.numel() for p in model.parameters())

        # Should be relatively lightweight (< 500k params)
        assert n_params < 500_000

    def test_eval_mode(self):
        model = PFCNN(num_classes=11)
        model.eval()

        x = torch.randn(4, 2, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        # Output should be deterministic in eval mode
        torch.testing.assert_close(out1, out2)

    def test_train_mode_dropout(self):
        model = PFCNN(num_classes=11, dropout=0.5)
        model.train()

        x = torch.randn(4, 2, 128)
        out1 = model(x)
        out2 = model(x)

        # Outputs may differ due to dropout (not guaranteed but likely)
        # This is a weak test, mainly checking it runs without error
        assert out1.shape == out2.shape
