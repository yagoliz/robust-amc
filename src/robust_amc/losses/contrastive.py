"""Contrastive loss functions for self-supervised learning.

This module implements the NT-Xent (Normalized Temperature-scaled Cross Entropy)
loss used in SimCLR and other contrastive learning frameworks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Also known as InfoNCE loss, this is the contrastive loss used in SimCLR.
    For a batch of N samples, each with 2 augmented views (2N total),
    it treats the two views of the same sample as positive pairs and
    all other samples as negatives.

    The loss is computed as:
        L = -log(exp(sim(z_i, z_j)/tau) / sum_k(exp(sim(z_i, z_k)/tau)))

    where z_i and z_j are embeddings of the two views of the same sample,
    and tau is the temperature parameter.

    Args:
        temperature: Temperature parameter for scaling similarities.
                     Lower temperature makes the model more confident.
                     Default is 0.5 (common choice from SimCLR).
        normalize: Whether to L2-normalize embeddings before computing loss.
                   Default True (recommended for stable training).
    """

    def __init__(self, temperature: float = 0.5, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent loss for two views.

        Args:
            z_i: Embeddings from first augmented view, shape (batch_size, embedding_dim)
            z_j: Embeddings from second augmented view, shape (batch_size, embedding_dim)

        Returns:
            Scalar loss value
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Normalize embeddings
        if self.normalize:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        # Concatenate embeddings: [z_i; z_j] -> shape (2*batch_size, embedding_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix: sim[i,j] = z[i] @ z[j]
        # Shape: (2*batch_size, 2*batch_size)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Create masks for positive pairs
        # For sample i (in first half), positive is at i + batch_size
        # For sample i (in second half), positive is at i - batch_size
        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # Create labels: positive pairs
        # First half: positives are at indices [batch_size, batch_size+1, ..., 2*batch_size-1]
        # Second half: positives are at indices [0, 1, ..., batch_size-1]
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ])

        # Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(temperature={self.temperature}, "
            f"normalize={self.normalize})"
        )


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Extends NT-Xent to incorporate label information. Samples with the same
    label are treated as positive pairs, even across different images.

    This can improve performance when labels are available during pretraining.

    Args:
        temperature: Temperature parameter for scaling similarities.
        normalize: Whether to L2-normalize embeddings.
        base_temperature: Base temperature for normalization (default 0.07).
    """

    def __init__(
        self,
        temperature: float = 0.5,
        normalize: bool = True,
        base_temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.base_temperature = base_temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            z_i: Embeddings from first augmented view, shape (batch_size, embedding_dim)
            z_j: Embeddings from second augmented view, shape (batch_size, embedding_dim)
            labels: Class labels, shape (batch_size,)

        Returns:
            Scalar loss value
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Normalize embeddings
        if self.normalize:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        # Concatenate embeddings and labels
        z = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Mask for positive pairs (same label, but not self)
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        self_mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        positive_mask = label_mask & ~self_mask

        # Count positives for each sample
        n_positives = positive_mask.sum(dim=1).float()

        # Mask out self-similarities with large negative number (not -inf for stability)
        large_neg = -1e9
        sim_matrix_masked = sim_matrix.masked_fill(self_mask, large_neg)

        # Use log_softmax for numerical stability
        log_prob = F.log_softmax(sim_matrix_masked, dim=1)

        # Mean of positive pair log probabilities
        # Only include samples that have at least one positive
        mask_valid = n_positives > 0

        if mask_valid.sum() == 0:
            # No valid samples with positives - return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute loss only for valid samples
        positive_log_prob = (positive_mask.float() * log_prob).sum(dim=1)
        n_positives_safe = n_positives.clamp(min=1)
        mean_log_prob = positive_log_prob / n_positives_safe

        # Scale by temperature ratio
        loss = -(self.base_temperature / self.temperature) * mean_log_prob

        # Only average over samples with positives
        return loss[mask_valid].mean()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(temperature={self.temperature}, "
            f"normalize={self.normalize})"
        )


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Projects encoder features to a lower-dimensional space where the
    contrastive loss is computed. This is a key component that improves
    the quality of learned representations.

    Args:
        input_dim: Dimension of input features from encoder.
        hidden_dim: Dimension of hidden layer.
        output_dim: Dimension of output projections.
        num_layers: Number of MLP layers (2 or 3 recommended).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Final layer (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
