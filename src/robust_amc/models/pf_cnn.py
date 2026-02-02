"""Phase-Feature CNN (PF-CNN) for modulation classification.

Based on the thesis architecture with dual-branch processing of
amplitude and phase features from I/Q signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FeatureBranch(nn.Module):
    """Single branch for processing either amplitude or phase features.

    Args:
        in_channels: Number of input channels (1 for amp/phase)
        n_filters: Base number of filters (np in thesis)
        n_stages: Number of convolutional stages (ns in thesis)
        seq_len: Input sequence length
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 4,
        n_stages: int = 5,
        seq_len: int = 128,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels

        for i in range(n_stages):
            out_channels = n_filters * (2 ** i)
            layers.append(ConvBlock(current_channels, out_channels))
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len)
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.squeeze(-1)  # (batch, output_dim)
        return x


class PFCNN(nn.Module):
    """Phase-Feature CNN with dual-branch architecture.

    Processes I/Q signals by extracting amplitude and phase,
    then processing each through parallel CNN branches.

    Args:
        num_classes: Number of modulation classes
        n_filters: Base number of filters per branch (np)
        n_stages: Number of convolutional stages per branch (ns)
        seq_len: Input sequence length
        hidden_dim: Hidden dimension in classifier
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 11,
        n_filters: int = 4,
        n_stages: int = 5,
        seq_len: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Amplitude branch
        self.amp_branch = FeatureBranch(
            in_channels=1,
            n_filters=n_filters,
            n_stages=n_stages,
            seq_len=seq_len,
        )

        # Phase branch
        self.phase_branch = FeatureBranch(
            in_channels=1,
            n_filters=n_filters,
            n_stages=n_stages,
            seq_len=seq_len,
        )

        # Combined feature dimension
        combined_dim = self.amp_branch.output_dim + self.phase_branch.output_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract amplitude and phase from I/Q signal.

        Args:
            x: Input tensor of shape (batch, 2, seq_len) where
               x[:, 0, :] is I (in-phase) and x[:, 1, :] is Q (quadrature)

        Returns:
            amplitude: (batch, 1, seq_len)
            phase: (batch, 1, seq_len)
        """
        i_signal = x[:, 0, :]  # (batch, seq_len)
        q_signal = x[:, 1, :]  # (batch, seq_len)

        # Amplitude: |I + jQ| = sqrt(I^2 + Q^2)
        amplitude = torch.sqrt(i_signal ** 2 + q_signal ** 2 + 1e-8)

        # Phase: atan2(Q, I)
        phase = torch.atan2(q_signal, i_signal)

        # Add channel dimension
        amplitude = amplitude.unsqueeze(1)  # (batch, 1, seq_len)
        phase = phase.unsqueeze(1)  # (batch, 1, seq_len)

        return amplitude, phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 2, seq_len)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Extract amplitude and phase
        amplitude, phase = self.extract_features(x)

        # Process through branches
        amp_features = self.amp_branch(amplitude)
        phase_features = self.phase_branch(phase)

        # Concatenate features
        combined = torch.cat([amp_features, phase_features], dim=1)

        # Classify
        logits = self.classifier(combined)

        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings before classification head.

        Useful for visualization and analysis.

        Args:
            x: Input tensor of shape (batch, 2, seq_len)

        Returns:
            Embeddings of shape (batch, combined_dim)
        """
        amplitude, phase = self.extract_features(x)
        amp_features = self.amp_branch(amplitude)
        phase_features = self.phase_branch(phase)
        return torch.cat([amp_features, phase_features], dim=1)


def create_pfcnn(
    num_classes: int = 11,
    variant: str = "default",
) -> PFCNN:
    """Factory function to create PF-CNN variants.

    Args:
        num_classes: Number of output classes
        variant: Model variant ('default', 'small', 'large')

    Returns:
        PFCNN model instance
    """
    configs = {
        "default": {"n_filters": 4, "n_stages": 5, "hidden_dim": 128, "dropout": 0.2},
        "small": {"n_filters": 2, "n_stages": 4, "hidden_dim": 64, "dropout": 0.1},
        "large": {"n_filters": 8, "n_stages": 6, "hidden_dim": 256, "dropout": 0.3},
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

    return PFCNN(num_classes=num_classes, **configs[variant])
