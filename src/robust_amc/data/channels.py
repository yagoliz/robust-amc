"""Channel models for simulating wireless propagation effects.

These transforms simulate the effects of wireless channels on I/Q signals,
including noise, multipath fading, and other propagation effects.

All channels follow the transform interface: `__call__(x) -> x`
where x has shape (2, seq_len) with x[0] = I and x[1] = Q.
"""

from typing import Optional

import numpy as np
import torch


class AWGN:
    """Additive White Gaussian Noise channel.

    Adds complex Gaussian noise to achieve a target SNR.

    Args:
        snr_db: Target signal-to-noise ratio in dB
        seed: Optional random seed for reproducibility
    """

    def __init__(self, snr_db: float, seed: Optional[int] = None):
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        # Calculate signal power
        signal_power = np.mean(x_np[0] ** 2 + x_np[1] ** 2)

        # Calculate noise power from SNR
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate complex noise (split power equally between I and Q)
        noise_std = np.sqrt(noise_power / 2)
        noise = self.rng.standard_normal(x_np.shape).astype(np.float32) * noise_std

        result = x_np + noise

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(snr_db={self.snr_db})"


class RayleighFading:
    """Rayleigh flat fading channel.

    Models flat fading from multiple scattered paths with no line-of-sight.
    The channel coefficient h is complex Gaussian with Rayleigh-distributed
    magnitude: h = h_r + j*h_i where h_r, h_i ~ N(0, sigma^2).

    This simulates a single-tap fading channel (flat fading assumption).

    Args:
        sigma: Scale parameter for the Rayleigh distribution (default 1.0
               gives unit average power E[|h|^2] = 2*sigma^2 = 1)
        seed: Optional random seed for reproducibility
    """

    def __init__(self, sigma: float = 1 / np.sqrt(2), seed: Optional[int] = None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        # Generate complex channel coefficient
        h_r = self.rng.standard_normal() * self.sigma
        h_i = self.rng.standard_normal() * self.sigma

        # Apply complex multiplication: y = h * x
        # (h_r + j*h_i)(I + j*Q) = (h_r*I - h_i*Q) + j(h_r*Q + h_i*I)
        I, Q = x_np[0], x_np[1]
        y_I = h_r * I - h_i * Q
        y_Q = h_r * Q + h_i * I

        result = np.stack([y_I, y_Q], axis=0).astype(np.float32)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma:.4f})"


class RicianFading:
    """Rician flat fading channel.

    Models flat fading with a dominant line-of-sight (LOS) component plus
    scattered paths. The K-factor is the ratio of LOS power to scattered power.

    h = sqrt(K/(K+1)) * exp(j*phi) + sqrt(1/(K+1)) * (h_r + j*h_i)

    where phi is the LOS phase and (h_r, h_i) are the scattered components.

    Args:
        k_factor: Rician K-factor (ratio of LOS to scattered power).
                  K=0 reduces to Rayleigh, K->inf is AWGN-like.
        los_phase: Phase of the line-of-sight component in radians.
                   If None, random phase is used.
        seed: Optional random seed for reproducibility
    """

    def __init__(
        self,
        k_factor: float = 1.0,
        los_phase: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.k_factor = k_factor
        self.los_phase = los_phase
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        K = self.k_factor

        # LOS component
        if self.los_phase is None:
            phi = self.rng.uniform(0, 2 * np.pi)
        else:
            phi = self.los_phase

        los_amp = np.sqrt(K / (K + 1))
        los_r = los_amp * np.cos(phi)
        los_i = los_amp * np.sin(phi)

        # Scattered component (Rayleigh with reduced power)
        scatter_amp = np.sqrt(1 / (K + 1))
        h_r = self.rng.standard_normal() * scatter_amp / np.sqrt(2)
        h_i = self.rng.standard_normal() * scatter_amp / np.sqrt(2)

        # Total channel coefficient
        h_total_r = los_r + h_r
        h_total_i = los_i + h_i

        # Apply complex multiplication
        I, Q = x_np[0], x_np[1]
        y_I = h_total_r * I - h_total_i * Q
        y_Q = h_total_r * Q + h_total_i * I

        result = np.stack([y_I, y_Q], axis=0).astype(np.float32)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k_factor={self.k_factor}, los_phase={self.los_phase})"


class TimeVaryingRayleigh:
    """Time-varying Rayleigh fading channel.

    Models a fading channel where the channel coefficient varies over time
    according to a Doppler spectrum (Jakes model). This creates time-selective
    fading effects.

    Args:
        doppler_hz: Maximum Doppler frequency in Hz
        sample_rate: Signal sample rate in Hz
        seed: Optional random seed for reproducibility
    """

    def __init__(
        self,
        doppler_hz: float = 10.0,
        sample_rate: float = 1e6,
        seed: Optional[int] = None,
    ):
        self.doppler_hz = doppler_hz
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        n_samples = x_np.shape[1]

        # Generate time-varying channel using sum-of-sinusoids (Jakes model)
        # Simplified: use filtered Gaussian process
        t = np.arange(n_samples) / self.sample_rate

        # Normalized Doppler frequency
        fd_norm = self.doppler_hz / self.sample_rate

        # Generate correlated fading using low-pass filtered noise
        # Correlation time ~ 1 / (2 * pi * fd)
        n_sinusoids = 8
        h_r = np.zeros(n_samples, dtype=np.float32)
        h_i = np.zeros(n_samples, dtype=np.float32)

        for _ in range(n_sinusoids):
            alpha = self.rng.uniform(0, 2 * np.pi)
            theta = self.rng.uniform(0, 2 * np.pi)
            f = self.doppler_hz * np.cos(alpha)
            h_r += np.cos(2 * np.pi * f * t + theta)
            h_i += np.sin(2 * np.pi * f * t + theta)

        # Normalize to unit average power
        h_r /= np.sqrt(n_sinusoids)
        h_i /= np.sqrt(n_sinusoids)

        # Apply time-varying complex multiplication
        I, Q = x_np[0], x_np[1]
        y_I = h_r * I - h_i * Q
        y_Q = h_r * Q + h_i * I

        result = np.stack([y_I, y_Q], axis=0).astype(np.float32)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(doppler_hz={self.doppler_hz}, sample_rate={self.sample_rate})"