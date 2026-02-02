"""Data transforms and normalization for I/Q signals."""

from typing import Callable

import numpy as np
import torch


class Compose:
    """Compose multiple transforms together.

    Args:
        transforms: List of transforms to apply in sequence
    """

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + "("]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append(")")
        return "\n".join(lines)


class ToTensor:
    """Convert numpy array to PyTorch tensor."""

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(x).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PowerNormalize:
    """Normalize signal to unit power.

    Computes: x_norm = x / sqrt(mean(I^2 + Q^2))

    This is the recommended normalization for I/Q signals as it
    preserves the relative phase information while normalizing amplitude.

    Args:
        eps: Small constant for numerical stability
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(x, torch.Tensor):
            power = torch.mean(x[0] ** 2 + x[1] ** 2)
            return x / torch.sqrt(power + self.eps)
        else:
            power = np.mean(x[0] ** 2 + x[1] ** 2)
            return x / np.sqrt(power + self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps})"


class Normalize:
    """Normalize I and Q channels independently to zero mean and unit variance.

    Computes for each channel c:
        x_norm[c] = (x[c] - mean(x[c])) / std(x[c])

    Args:
        eps: Small constant for numerical stability
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_out = x.clone()
            for c in range(2):
                mean = x_out[c].mean()
                std = x_out[c].std()
                x_out[c] = (x_out[c] - mean) / (std + self.eps)
            return x_out
        else:
            x_out = x.copy()
            for c in range(2):
                mean = x_out[c].mean()
                std = x_out[c].std()
                x_out[c] = (x_out[c] - mean) / (std + self.eps)
            return x_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps})"


class ComplexToIQ:
    """Convert complex-valued signal to I/Q representation.

    Input: complex array of shape (N,)
    Output: real array of shape (2, N) where [0] is I and [1] is Q
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(x):
            return np.stack([x.real, x.imag], axis=0).astype(np.float32)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class IQToComplex:
    """Convert I/Q representation to complex-valued signal.

    Input: real array of shape (2, N) where [0] is I and [1] is Q
    Output: complex array of shape (N,)
    """

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return x[0] + 1j * x[1]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_default_transform() -> Compose:
    """Get default transform pipeline for evaluation.

    Returns:
        Compose transform with power normalization and tensor conversion
    """
    return Compose([
        PowerNormalize(),
        ToTensor(),
    ])


def get_train_transform() -> Compose:
    """Get default transform pipeline for training.

    Note: Augmentations (MDA-DMC) should be added separately.

    Returns:
        Compose transform with power normalization and tensor conversion
    """
    return Compose([
        PowerNormalize(),
        ToTensor(),
    ])
