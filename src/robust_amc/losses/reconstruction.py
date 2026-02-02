"""Reconstruction losses and decoder architectures for self-supervised learning.

This module implements the self-reconstruction component of CLSR-AMC,
which helps the encoder learn meaningful signal representations by
requiring it to reconstruct the original I/Q signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """Combined reconstruction loss for I/Q signal reconstruction.

    Supports multiple loss components:
    - MSE: Mean squared error between reconstructed and original signal
    - Amplitude MSE: MSE on signal amplitude (|I + jQ|)
    - Phase MSE: MSE on signal phase (atan2(Q, I))

    Args:
        mse_weight: Weight for MSE loss component.
        amplitude_weight: Weight for amplitude reconstruction loss.
        phase_weight: Weight for phase reconstruction loss.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        amplitude_weight: float = 0.0,
        phase_weight: float = 0.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.amplitude_weight = amplitude_weight
        self.phase_weight = phase_weight

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            reconstructed: Reconstructed signal, shape (batch, 2, seq_len)
            original: Original signal, shape (batch, 2, seq_len)

        Returns:
            Scalar loss value
        """
        loss = 0.0

        # MSE on I/Q directly
        if self.mse_weight > 0:
            mse_loss = F.mse_loss(reconstructed, original)
            loss = loss + self.mse_weight * mse_loss

        # Amplitude reconstruction
        if self.amplitude_weight > 0:
            orig_amp = torch.sqrt(original[:, 0] ** 2 + original[:, 1] ** 2 + 1e-8)
            recon_amp = torch.sqrt(reconstructed[:, 0] ** 2 + reconstructed[:, 1] ** 2 + 1e-8)
            amp_loss = F.mse_loss(recon_amp, orig_amp)
            loss = loss + self.amplitude_weight * amp_loss

        # Phase reconstruction (with wrapping handling)
        if self.phase_weight > 0:
            orig_phase = torch.atan2(original[:, 1], original[:, 0])
            recon_phase = torch.atan2(reconstructed[:, 1], reconstructed[:, 0])
            # Phase difference with wrapping
            phase_diff = torch.atan2(
                torch.sin(recon_phase - orig_phase),
                torch.cos(recon_phase - orig_phase),
            )
            phase_loss = (phase_diff ** 2).mean()
            loss = loss + self.phase_weight * phase_loss

        return loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mse_weight={self.mse_weight}, "
            f"amplitude_weight={self.amplitude_weight}, "
            f"phase_weight={self.phase_weight})"
        )


class SignalDecoder(nn.Module):
    """Decoder network for reconstructing I/Q signals from embeddings.

    Uses transposed convolutions to upsample the embedding back to the
    original signal shape.

    Args:
        embedding_dim: Dimension of input embeddings.
        hidden_dims: List of hidden channel dimensions.
        output_len: Length of output signal (default 128).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: list[int] = None,
        output_len: int = 128,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        self.output_len = output_len

        # Initial projection from embedding to feature maps
        # We'll reshape to (batch, hidden_dims[0], initial_len)
        initial_len = output_len // (2 ** len(hidden_dims))
        self.initial_len = max(initial_len, 1)

        self.fc = nn.Linear(embedding_dim, hidden_dims[0] * self.initial_len)
        self.initial_channels = hidden_dims[0]

        # Transposed convolution layers for upsampling
        layers = []
        for i in range(len(hidden_dims) - 1):
            in_ch = hidden_dims[i]
            out_ch = hidden_dims[i + 1]

            layers.append(
                nn.ConvTranspose1d(
                    in_ch, out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))

        # Final layer to output I/Q channels
        layers.append(
            nn.ConvTranspose1d(
                hidden_dims[-1], 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings back to I/Q signal.

        Args:
            z: Embeddings of shape (batch, embedding_dim)

        Returns:
            Reconstructed signal of shape (batch, 2, output_len)
        """
        batch_size = z.size(0)

        # Project to initial feature map
        x = self.fc(z)
        x = x.view(batch_size, self.initial_channels, self.initial_len)

        # Upsample through decoder
        x = self.decoder(x)

        # Ensure output length matches
        if x.size(2) != self.output_len:
            x = F.interpolate(x, size=self.output_len, mode="linear", align_corners=False)

        return x


class LightweightDecoder(nn.Module):
    """Lightweight decoder using only linear layers.

    A simpler alternative to the convolutional decoder, useful when
    reconstruction quality is less critical or for faster training.

    Args:
        embedding_dim: Dimension of input embeddings.
        hidden_dim: Dimension of hidden layer.
        output_len: Length of output signal (default 128).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        output_len: int = 128,
    ):
        super().__init__()
        self.output_len = output_len

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * output_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings back to I/Q signal.

        Args:
            z: Embeddings of shape (batch, embedding_dim)

        Returns:
            Reconstructed signal of shape (batch, 2, output_len)
        """
        x = self.decoder(z)
        return x.view(-1, 2, self.output_len)