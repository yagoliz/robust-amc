"""MDA-DMC data augmentation techniques for robust modulation classification.

This module implements the Multi-Domain Augmentation for Domain-Mismatch
Compensation (MDA-DMC) techniques described in the thesis:

1. AGN: Additive Gaussian Noise with SNR jitter
2. RSC: Rotation in Signal Constellation (I/Q plane rotation)
3. SSC: Stretching in Signal Constellation (amplitude scaling)
4. RSSC: Combined Rotation and Stretching

All augmentations follow the transform interface: `__call__(x) -> x`
where x has shape (2, seq_len) with x[0] = I and x[1] = Q.

References:
    - PhD Thesis Chapter on Data Augmentation
    - "Deep Learning-Based Signal Classification for Automated Modulation Recognition"
"""

from typing import Optional, Callable

import numpy as np
import torch


class AdditiveGaussianNoise:
    """Additive Gaussian Noise (AGN) augmentation with SNR jitter.

    Adds Gaussian noise to the signal with a randomly selected SNR value
    from a specified range. This helps the model generalize across
    different noise conditions.

    The noise is added as:
        y = x + noise
    where noise has power determined by the target SNR.

    Args:
        snr_range: Tuple of (min_snr_db, max_snr_db) for random SNR selection.
                   Default is (-5, 15) dB which covers typical operating range.
        p: Probability of applying the augmentation. Default 0.5.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        snr_range: tuple[float, float] = (-5.0, 15.0),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.snr_range = snr_range
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.rng.random() > self.p:
            return x

        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x.copy()

        # Calculate signal power
        signal_power = np.mean(x_np[0] ** 2 + x_np[1] ** 2)

        # Random SNR in range
        snr_db = self.rng.uniform(self.snr_range[0], self.snr_range[1])
        snr_linear = 10 ** (snr_db / 10)

        # Calculate noise power
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)  # Split between I and Q

        # Generate and add noise
        noise_i = self.rng.standard_normal(x_np.shape[1]).astype(np.float32) * noise_std
        noise_q = self.rng.standard_normal(x_np.shape[1]).astype(np.float32) * noise_std

        result = np.stack([
            x_np[0] + noise_i,
            x_np[1] + noise_q,
        ], axis=0)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(snr_range={self.snr_range}, p={self.p})"


class RotationInSignalConstellation:
    """Rotation in Signal Constellation (RSC) augmentation.

    Rotates the I/Q signal by a random angle in the complex plane.
    This simulates carrier phase offset and helps the model become
    invariant to absolute phase.

    The rotation is applied as:
        y_I = I * cos(theta) - Q * sin(theta)
        y_Q = I * sin(theta) + Q * cos(theta)

    Args:
        angle_range: Tuple of (min_angle, max_angle) in degrees.
                     Default is (-180, 180) for full rotation range.
        p: Probability of applying the augmentation. Default 0.5.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        angle_range: tuple[float, float] = (-180.0, 180.0),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.angle_range = angle_range
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.rng.random() > self.p:
            return x

        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        # Random rotation angle
        angle_deg = self.rng.uniform(self.angle_range[0], self.angle_range[1])
        angle_rad = np.deg2rad(angle_deg)

        cos_theta = np.cos(angle_rad).astype(np.float32)
        sin_theta = np.sin(angle_rad).astype(np.float32)

        I, Q = x_np[0], x_np[1]
        y_I = I * cos_theta - Q * sin_theta
        y_Q = I * sin_theta + Q * cos_theta

        result = np.stack([y_I, y_Q], axis=0)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angle_range={self.angle_range}, p={self.p})"


class StretchingInSignalConstellation:
    """Stretching in Signal Constellation (SSC) augmentation.

    Scales the I/Q signal amplitude by a random factor. This simulates
    automatic gain control (AGC) variations and amplitude mismatches.

    The scaling is applied as:
        y = scale * x

    Args:
        scale_range: Tuple of (min_scale, max_scale) as linear multipliers.
                     Default is (0.8, 1.2) for +/- 20% amplitude variation.
        p: Probability of applying the augmentation. Default 0.5.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.scale_range = scale_range
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.rng.random() > self.p:
            return x

        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        # Random scaling factor
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])
        scale = np.float32(scale)

        result = (x_np * scale).astype(np.float32)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_range={self.scale_range}, p={self.p})"


class RotationAndStretchingInSignalConstellation:
    """Combined Rotation and Stretching (RSSC) augmentation.

    Applies both rotation and scaling to the I/Q signal. This provides
    a more comprehensive augmentation that combines the benefits of
    both RSC and SSC.

    The transform is applied as:
        y_I = scale * (I * cos(theta) - Q * sin(theta))
        y_Q = scale * (I * sin(theta) + Q * cos(theta))

    Args:
        angle_range: Tuple of (min_angle, max_angle) in degrees.
        scale_range: Tuple of (min_scale, max_scale) as linear multipliers.
        p: Probability of applying the augmentation. Default 0.5.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        angle_range: tuple[float, float] = (-180.0, 180.0),
        scale_range: tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.rng.random() > self.p:
            return x

        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        # Random rotation angle
        angle_deg = self.rng.uniform(self.angle_range[0], self.angle_range[1])
        angle_rad = np.deg2rad(angle_deg)

        # Random scaling factor
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])

        cos_theta = np.float32(np.cos(angle_rad) * scale)
        sin_theta = np.float32(np.sin(angle_rad) * scale)

        I, Q = x_np[0], x_np[1]
        y_I = I * cos_theta - Q * sin_theta
        y_Q = I * sin_theta + Q * cos_theta

        result = np.stack([y_I, y_Q], axis=0)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(angle_range={self.angle_range}, "
            f"scale_range={self.scale_range}, p={self.p})"
        )


class TimeShift:
    """Time/sample shift augmentation.

    Circular shifts the signal by a random number of samples.
    This helps the model become invariant to the absolute timing offset.

    Args:
        max_shift: Maximum shift in samples (positive or negative).
                   Default is 16 samples.
        p: Probability of applying the augmentation. Default 0.5.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        max_shift: int = 16,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.max_shift = max_shift
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.rng.random() > self.p:
            return x

        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        # Random shift amount
        shift = self.rng.integers(-self.max_shift, self.max_shift + 1)

        # Apply circular shift to both I and Q
        result = np.stack([
            np.roll(x_np[0], shift),
            np.roll(x_np[1], shift),
        ], axis=0)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_shift={self.max_shift}, p={self.p})"


class RandomFlip:
    """Random flip augmentation.

    Randomly flips the signal in I, Q, or both dimensions.
    This helps the model learn symmetry properties.

    Args:
        flip_i: Whether to randomly flip I channel. Default True.
        flip_q: Whether to randomly flip Q channel. Default True.
        p: Probability of flipping each enabled channel. Default 0.5.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        flip_i: bool = True,
        flip_q: bool = True,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.flip_i = flip_i
        self.flip_q = flip_q
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy().copy()
        else:
            x_np = x.copy()

        # Randomly flip I
        if self.flip_i and self.rng.random() < self.p:
            x_np[0] = -x_np[0]

        # Randomly flip Q
        if self.flip_q and self.rng.random() < self.p:
            x_np[1] = -x_np[1]

        if is_tensor:
            return torch.from_numpy(x_np)
        return x_np

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flip_i={self.flip_i}, flip_q={self.flip_q}, p={self.p})"


class RandomAugmentation:
    """Randomly apply one of several augmentations.

    Selects one augmentation from a list with equal probability and applies it.
    Useful for combining multiple augmentation strategies.

    Args:
        augmentations: List of augmentation transforms.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        augmentations: list[Callable],
        seed: Optional[int] = None,
    ):
        self.augmentations = augmentations
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        idx = self.rng.integers(0, len(self.augmentations))
        return self.augmentations[idx](x)

    def __repr__(self) -> str:
        aug_names = [a.__class__.__name__ for a in self.augmentations]
        return f"{self.__class__.__name__}(augmentations={aug_names})"


class MDADMCPipeline:
    """Complete MDA-DMC augmentation pipeline.

    Composes multiple MDA-DMC augmentations into a single transform.
    Each augmentation is applied with its own probability.

    Args:
        agn: Enable AGN (noise jitter). Default True.
        rsc: Enable RSC (rotation). Default True.
        ssc: Enable SSC (scaling). Default True.
        time_shift: Enable time shift. Default False.
        random_flip: Enable random flip. Default False.
        agn_snr_range: SNR range for AGN in dB.
        rsc_angle_range: Angle range for RSC in degrees.
        ssc_scale_range: Scale range for SSC.
        time_shift_max: Maximum samples for time shift.
        p: Base probability for each augmentation. Default 0.5.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        agn: bool = True,
        rsc: bool = True,
        ssc: bool = True,
        time_shift: bool = False,
        random_flip: bool = False,
        agn_snr_range: tuple[float, float] = (-5.0, 15.0),
        rsc_angle_range: tuple[float, float] = (-180.0, 180.0),
        ssc_scale_range: tuple[float, float] = (0.8, 1.2),
        time_shift_max: int = 16,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.p = p
        self.transforms = []

        # Create child RNGs for reproducibility
        base_rng = np.random.default_rng(seed)

        if agn:
            self.transforms.append(
                AdditiveGaussianNoise(
                    snr_range=agn_snr_range,
                    p=p,
                    seed=int(base_rng.integers(0, 2**31)),
                )
            )

        if rsc:
            self.transforms.append(
                RotationInSignalConstellation(
                    angle_range=rsc_angle_range,
                    p=p,
                    seed=int(base_rng.integers(0, 2**31)),
                )
            )

        if ssc:
            self.transforms.append(
                StretchingInSignalConstellation(
                    scale_range=ssc_scale_range,
                    p=p,
                    seed=int(base_rng.integers(0, 2**31)),
                )
            )

        if time_shift:
            self.transforms.append(
                TimeShift(
                    max_shift=time_shift_max,
                    p=p,
                    seed=int(base_rng.integers(0, 2**31)),
                )
            )

        if random_flip:
            self.transforms.append(
                RandomFlip(
                    p=p,
                    seed=int(base_rng.integers(0, 2**31)),
                )
            )

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(p={self.p}, transforms=["]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append("])")
        return "\n".join(lines)


# Convenience aliases
AGN = AdditiveGaussianNoise
RSC = RotationInSignalConstellation
SSC = StretchingInSignalConstellation
RSSC = RotationAndStretchingInSignalConstellation