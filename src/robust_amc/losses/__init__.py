"""Loss functions for training."""

from .contrastive import NTXentLoss, SupConLoss, ProjectionHead
from .reconstruction import ReconstructionLoss, SignalDecoder, LightweightDecoder

__all__ = [
    # Contrastive losses
    "NTXentLoss",
    "SupConLoss",
    "ProjectionHead",
    # Reconstruction losses
    "ReconstructionLoss",
    "SignalDecoder",
    "LightweightDecoder",
]