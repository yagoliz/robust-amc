"""CLSR-AMC: Contrastive Learning with Self-Reconstruction for AMC.

This module implements the CLSR-AMC architecture which combines:
1. Contrastive learning to learn robust representations
2. Self-reconstruction to preserve signal information
3. Classification for modulation recognition

The model can be trained in multiple modes:
- Supervised: Classification loss only (baseline)
- Contrastive: NT-Xent loss for unsupervised pretraining
- Multi-task: Combined contrastive + reconstruction + classification
"""

import torch
import torch.nn as nn

from .pf_cnn import FeatureBranch
from ..losses.contrastive import ProjectionHead
from ..losses.reconstruction import SignalDecoder, LightweightDecoder


class CLSRAMCEncoder(nn.Module):
    """Encoder for CLSR-AMC using PF-CNN style architecture.

    Processes I/Q signals using the same dual-branch (amplitude/phase)
    architecture as PF-CNN but with a configurable output.

    Args:
        n_filters: Base number of filters per branch.
        n_stages: Number of convolutional stages per branch.
        seq_len: Input sequence length.
        pool_output_size: Output size of adaptive average pooling per branch.
            >1 preserves coarse temporal structure for reconstruction.
    """

    def __init__(
        self,
        n_filters: int = 4,
        n_stages: int = 5,
        seq_len: int = 128,
        pool_output_size: int = 1,
    ):
        super().__init__()

        # Amplitude branch
        self.amp_branch = FeatureBranch(
            in_channels=1,
            n_filters=n_filters,
            n_stages=n_stages,
            seq_len=seq_len,
            pool_output_size=pool_output_size,
        )

        # Phase branch
        self.phase_branch = FeatureBranch(
            in_channels=1,
            n_filters=n_filters,
            n_stages=n_stages,
            seq_len=seq_len,
            pool_output_size=pool_output_size,
        )

        # Output dimension is sum of both branches
        self.output_dim = self.amp_branch.output_dim + self.phase_branch.output_dim

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract amplitude and phase from I/Q signal."""
        i_signal = x[:, 0, :]
        q_signal = x[:, 1, :]

        amplitude = torch.sqrt(i_signal ** 2 + q_signal ** 2 + 1e-8)
        phase = torch.atan2(q_signal, i_signal)

        amplitude = amplitude.unsqueeze(1)
        phase = phase.unsqueeze(1)

        return amplitude, phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode I/Q signal to embeddings.

        Args:
            x: Input signal of shape (batch, 2, seq_len)

        Returns:
            Embeddings of shape (batch, output_dim)
        """
        amplitude, phase = self.extract_features(x)
        amp_features = self.amp_branch(amplitude)
        phase_features = self.phase_branch(phase)
        return torch.cat([amp_features, phase_features], dim=1)


class CLSRAMC(nn.Module):
    """CLSR-AMC: Contrastive Learning with Self-Reconstruction for AMC.

    A multi-task model that learns robust representations through:
    1. Contrastive learning between augmented views
    2. Self-reconstruction of the original signal
    3. Classification of modulation type

    Args:
        num_classes: Number of modulation classes.
        encoder_filters: Base filters for encoder.
        encoder_stages: Number of stages in encoder.
        seq_len: Input sequence length.
        projection_dim: Dimension of contrastive projection.
        hidden_dim: Hidden dimension in classifier.
        dropout: Dropout probability.
        decoder_type: Type of decoder ('conv' or 'linear').
        pool_output_size: Encoder pooling output size. >1 preserves temporal
            structure for reconstruction.
    """

    def __init__(
        self,
        num_classes: int = 11,
        encoder_filters: int = 4,
        encoder_stages: int = 5,
        seq_len: int = 128,
        projection_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        decoder_type: str = "linear",
        pool_output_size: int = 1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.seq_len = seq_len

        # Shared encoder
        self.encoder = CLSRAMCEncoder(
            n_filters=encoder_filters,
            n_stages=encoder_stages,
            seq_len=seq_len,
            pool_output_size=pool_output_size,
        )
        self.embedding_dim = self.encoder.output_dim

        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=self.embedding_dim,
            hidden_dim=256,
            output_dim=projection_dim,
            num_layers=2,
        )

        # Decoder for self-reconstruction
        if decoder_type == "conv":
            self.decoder = SignalDecoder(
                embedding_dim=self.embedding_dim,
                hidden_dims=[256, 128, 64, 32],
                output_len=seq_len,
            )
        else:
            self.decoder = LightweightDecoder(
                embedding_dim=self.embedding_dim,
                hidden_dim=512,
                output_len=seq_len,
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embeddings.

        Args:
            x: Input signal of shape (batch, 2, seq_len)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        return self.encoder(x)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project embeddings for contrastive loss.

        Args:
            z: Embeddings of shape (batch, embedding_dim)

        Returns:
            Projections of shape (batch, projection_dim)
        """
        return self.projection_head(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings back to signal.

        Args:
            z: Embeddings of shape (batch, embedding_dim)

        Returns:
            Reconstructed signal of shape (batch, 2, seq_len)
        """
        return self.decoder(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Classify from embeddings.

        Args:
            z: Embeddings of shape (batch, embedding_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.classifier(z)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input signal of shape (batch, 2, seq_len)
            return_all: If True, return dict with embeddings, projections,
                        reconstruction, and logits. Otherwise just logits.

        Returns:
            Logits or dict with all outputs
        """
        # Encode
        z = self.encode(x)

        if not return_all:
            # Just classification
            return self.classify(z)

        # Full forward pass
        p = self.project(z)
        x_recon = self.decode(z)
        logits = self.classify(z)

        return {
            "embeddings": z,
            "projections": p,
            "reconstruction": x_recon,
            "logits": logits,
        }

    def forward_contrastive(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for contrastive training with two views.

        Args:
            x_i: First augmented view, shape (batch, 2, seq_len)
            x_j: Second augmented view, shape (batch, 2, seq_len)

        Returns:
            Tuple of (z_i, z_j, p_i, p_j) where z are embeddings
            and p are projections
        """
        z_i = self.encode(x_i)
        z_j = self.encode(x_j)
        p_i = self.project(z_i)
        p_j = self.project(z_j)
        return z_i, z_j, p_i, p_j

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for visualization/analysis."""
        return self.encode(x)


class CLSRAMCLoss(nn.Module):
    """Combined loss for CLSR-AMC training.

    Combines:
    1. Contrastive loss (NT-Xent) between augmented views
    2. Reconstruction loss (MSE) on decoded signal
    3. Classification loss (CrossEntropy)

    When balance_losses=True, each loss component is normalized by its
    exponential moving average before weighting. This ensures all three
    objectives contribute equal gradient magnitude regardless of their
    raw scale (contrastive ~4.5 vs reconstruction ~0.5 vs classification ~1.2).

    Args:
        contrastive_weight: Weight for contrastive loss.
        reconstruction_weight: Weight for reconstruction loss.
        classification_weight: Weight for classification loss.
        temperature: Temperature for NT-Xent loss.
        balance_losses: Auto-balance loss magnitudes via EMA normalization.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        classification_weight: float = 1.0,
        temperature: float = 0.5,
        balance_losses: bool = True,
    ):
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight
        self.balance_losses = balance_losses

        from ..losses.contrastive import NTXentLoss
        from ..losses.reconstruction import ReconstructionLoss

        self.contrastive_loss = NTXentLoss(temperature=temperature)
        self.reconstruction_loss = ReconstructionLoss(mse_weight=1.0)
        self.classification_loss = nn.CrossEntropyLoss()

        # EMA trackers for loss balancing
        if balance_losses:
            self.register_buffer("_ema_con", torch.tensor(1.0))
            self.register_buffer("_ema_rec", torch.tensor(1.0))
            self.register_buffer("_ema_cls", torch.tensor(1.0))
            self.register_buffer("_ema_initialized", torch.tensor(False))
            self._ema_decay = 0.99

    def forward(
        self,
        p_i: torch.Tensor,
        p_j: torch.Tensor,
        x_recon: torch.Tensor,
        x_original: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            p_i: Projections from first view
            p_j: Projections from second view
            x_recon: Reconstructed signal
            x_original: Original signal
            logits: Classification logits
            labels: Ground truth labels

        Returns:
            Dict with total loss and individual components
        """
        losses = {}

        # Contrastive loss
        if self.contrastive_weight > 0:
            l_con = self.contrastive_loss(p_i, p_j)
            losses["contrastive"] = l_con
        else:
            l_con = torch.tensor(0.0, device=logits.device)
            losses["contrastive"] = l_con

        # Reconstruction loss
        if self.reconstruction_weight > 0:
            l_rec = self.reconstruction_loss(x_recon, x_original)
            losses["reconstruction"] = l_rec
        else:
            l_rec = torch.tensor(0.0, device=logits.device)
            losses["reconstruction"] = l_rec

        # Classification loss
        if self.classification_weight > 0:
            l_cls = self.classification_loss(logits, labels)
            losses["classification"] = l_cls
        else:
            l_cls = torch.tensor(0.0, device=logits.device)
            losses["classification"] = l_cls

        # Combined loss with optional auto-balancing
        # Only balance when training and all three components are active
        _do_balance = (
            self.balance_losses
            and self.training
            and self.contrastive_weight > 0
            and self.reconstruction_weight > 0
            and self.classification_weight > 0
        )

        if _do_balance:
            with torch.no_grad():
                if not self._ema_initialized:
                    # Initialize EMAs to current loss values
                    self._ema_con = l_con.detach().clamp(min=1e-6)
                    self._ema_rec = l_rec.detach().clamp(min=1e-6)
                    self._ema_cls = l_cls.detach().clamp(min=1e-6)
                    self._ema_initialized = torch.tensor(True)
                else:
                    d = self._ema_decay
                    self._ema_con = d * self._ema_con + (1 - d) * l_con.detach()
                    self._ema_rec = d * self._ema_rec + (1 - d) * l_rec.detach()
                    self._ema_cls = d * self._ema_cls + (1 - d) * l_cls.detach()

            # Normalize each loss by its EMA so all contribute ~1.0 before weighting
            total = (
                self.contrastive_weight * l_con / self._ema_con.detach().clamp(min=1e-6) +
                self.reconstruction_weight * l_rec / self._ema_rec.detach().clamp(min=1e-6) +
                self.classification_weight * l_cls / self._ema_cls.detach().clamp(min=1e-6)
            )
        else:
            total = (
                self.contrastive_weight * l_con +
                self.reconstruction_weight * l_rec +
                self.classification_weight * l_cls
            )

        losses["total"] = total

        return losses


def create_clsr_amc(
    num_classes: int = 11,
    variant: str = "default",
    seq_len: int = 128,
) -> CLSRAMC:
    """Factory function to create CLSR-AMC variants.

    Args:
        num_classes: Number of output classes.
        variant: Model variant ('default', 'small', 'large').
        seq_len: Input sequence length (must match data crop_length).

    Returns:
        CLSRAMC model instance.
    """
    configs = {
        "default": {
            "encoder_filters": 4,
            "encoder_stages": 5,
            "projection_dim": 128,
            "hidden_dim": 128,
            "dropout": 0.2,
            "decoder_type": "linear",
            "pool_output_size": 4,
        },
        "small": {
            "encoder_filters": 2,
            "encoder_stages": 4,
            "projection_dim": 64,
            "hidden_dim": 64,
            "dropout": 0.1,
            "decoder_type": "linear",
            "pool_output_size": 4,
        },
        "large": {
            "encoder_filters": 8,
            "encoder_stages": 6,
            "projection_dim": 256,
            "hidden_dim": 256,
            "dropout": 0.3,
            "decoder_type": "conv",
            "pool_output_size": 4,
        },
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

    return CLSRAMC(num_classes=num_classes, seq_len=seq_len, **configs[variant])
