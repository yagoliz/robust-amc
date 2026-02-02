"""Hardware impairment models for I/Q signals.

These transforms simulate common hardware imperfections in RF receivers,
including carrier frequency offset, I/Q imbalance, and DC offset.

All impairments follow the transform interface: `__call__(x) -> x`
where x has shape (2, seq_len) with x[0] = I and x[1] = Q.
"""

from typing import Optional

import numpy as np
import torch


class CarrierFrequencyOffset:
    """Carrier Frequency Offset (CFO) impairment.

    Simulates the effect of a frequency mismatch between transmitter and
    receiver oscillators. This causes a time-varying phase rotation:

        y(t) = x(t) * exp(j * 2 * pi * delta_f * t)

    Args:
        delta_f: Frequency offset in Hz (can be positive or negative)
        sample_rate: Signal sample rate in Hz
        initial_phase: Initial phase offset in radians (default 0)
        seed: Optional random seed for randomizing initial phase
    """

    def __init__(
        self,
        delta_f: float,
        sample_rate: float = 1e6,
        initial_phase: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.delta_f = delta_f
        self.sample_rate = sample_rate
        self.initial_phase = initial_phase
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        n_samples = x_np.shape[1]
        t = np.arange(n_samples) / self.sample_rate

        # Random initial phase if RNG is available
        if self.rng is not None:
            phase_offset = self.rng.uniform(0, 2 * np.pi)
        else:
            phase_offset = self.initial_phase

        # Phase rotation over time
        phase = 2 * np.pi * self.delta_f * t + phase_offset

        # Apply rotation: y = x * exp(j*phase)
        cos_phase = np.cos(phase).astype(np.float32)
        sin_phase = np.sin(phase).astype(np.float32)

        I, Q = x_np[0], x_np[1]
        y_I = I * cos_phase - Q * sin_phase
        y_Q = I * sin_phase + Q * cos_phase

        result = np.stack([y_I, y_Q], axis=0)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(delta_f={self.delta_f}, "
            f"sample_rate={self.sample_rate})"
        )


class IQImbalance:
    """I/Q Imbalance impairment.

    Simulates amplitude and phase mismatch between I and Q branches in
    a quadrature receiver. The impaired signal is:

        y_I = I
        y_Q = g * (cos(phi) * Q + sin(phi) * I)

    where g is the amplitude imbalance and phi is the phase imbalance.

    This is equivalent to the more symmetric formulation used in many papers:
        y = alpha * x + beta * conj(x)

    Args:
        amplitude_imbalance_db: Amplitude imbalance in dB (Q relative to I)
        phase_imbalance_deg: Phase imbalance in degrees
    """

    def __init__(
        self,
        amplitude_imbalance_db: float = 0.0,
        phase_imbalance_deg: float = 0.0,
    ):
        self.amplitude_imbalance_db = amplitude_imbalance_db
        self.phase_imbalance_deg = phase_imbalance_deg

        # Convert to linear values
        self.g = 10 ** (amplitude_imbalance_db / 20)
        self.phi = np.deg2rad(phase_imbalance_deg)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        I, Q = x_np[0], x_np[1]

        # Apply I/Q imbalance
        y_I = I
        y_Q = self.g * (np.cos(self.phi) * Q + np.sin(self.phi) * I)

        result = np.stack([y_I, y_Q], axis=0).astype(np.float32)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(amplitude_imbalance_db={self.amplitude_imbalance_db}, "
            f"phase_imbalance_deg={self.phase_imbalance_deg})"
        )


class DCOffset:
    """DC Offset impairment.

    Simulates DC offset in the receiver caused by LO leakage or ADC offset.
    Adds a constant offset to I and/or Q channels:

        y_I = I + dc_i
        y_Q = Q + dc_q

    Args:
        dc_i: DC offset for I channel (absolute value or fraction of signal std)
        dc_q: DC offset for Q channel (absolute value or fraction of signal std)
        relative: If True, dc_i and dc_q are multiplied by signal std
    """

    def __init__(
        self,
        dc_i: float = 0.0,
        dc_q: float = 0.0,
        relative: bool = False,
    ):
        self.dc_i = dc_i
        self.dc_q = dc_q
        self.relative = relative

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        if self.relative:
            # Scale DC offset by signal standard deviation
            scale = np.sqrt(np.mean(x_np[0] ** 2 + x_np[1] ** 2))
            dc_i = self.dc_i * scale
            dc_q = self.dc_q * scale
        else:
            dc_i = self.dc_i
            dc_q = self.dc_q

        y_I = x_np[0] + dc_i
        y_Q = x_np[1] + dc_q

        result = np.stack([y_I, y_Q], axis=0).astype(np.float32)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(dc_i={self.dc_i}, dc_q={self.dc_q}, "
            f"relative={self.relative})"
        )


class PhaseNoise:
    """Phase noise impairment.

    Simulates oscillator phase noise using a random walk model. This causes
    the signal phase to drift over time:

        y(t) = x(t) * exp(j * phi(t))

    where phi(t) is a Wiener process (integrated white noise).

    Args:
        std_per_sample: Standard deviation of phase increment per sample (radians)
        seed: Optional random seed for reproducibility
    """

    def __init__(self, std_per_sample: float = 0.01, seed: Optional[int] = None):
        self.std_per_sample = std_per_sample
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        n_samples = x_np.shape[1]

        # Generate phase noise using random walk
        phase_increments = self.rng.standard_normal(n_samples) * self.std_per_sample
        phase = np.cumsum(phase_increments).astype(np.float32)

        # Apply phase rotation
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        I, Q = x_np[0], x_np[1]
        y_I = I * cos_phase - Q * sin_phase
        y_Q = I * sin_phase + Q * cos_phase

        result = np.stack([y_I, y_Q], axis=0)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std_per_sample={self.std_per_sample})"


class SampleRateOffset:
    """Sample Rate Offset (SRO) impairment.

    Simulates timing drift caused by mismatch between transmitter and
    receiver sampling clocks. Uses linear interpolation to resample
    the signal at a slightly different rate.

    Args:
        ppm: Clock offset in parts per million (positive = faster receiver clock)
    """

    def __init__(self, ppm: float = 0.0):
        self.ppm = ppm

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x

        n_samples = x_np.shape[1]

        # Original sample indices
        original_indices = np.arange(n_samples)

        # New sample indices (with offset)
        rate_factor = 1 + self.ppm * 1e-6
        new_indices = original_indices * rate_factor

        # Clip to valid range and interpolate
        new_indices = np.clip(new_indices, 0, n_samples - 1)

        # Linear interpolation for I and Q
        y_I = np.interp(original_indices, new_indices, x_np[0]).astype(np.float32)
        y_Q = np.interp(original_indices, new_indices, x_np[1]).astype(np.float32)

        result = np.stack([y_I, y_Q], axis=0)

        if is_tensor:
            return torch.from_numpy(result)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ppm={self.ppm})"