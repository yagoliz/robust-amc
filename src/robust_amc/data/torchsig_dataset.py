"""TorchSig dataset loader with family mapping and impairment configurations.

This module provides a PyTorch Dataset wrapper for TorchSig synthetic signals
with support for:
- Modulation family mapping (PSK, FSK, AM, SSB, QAM)
- Configurable impairment levels (train vs OOD)
- Random cropping for input size flexibility
- Caching for efficient reuse
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .label_mapping import FamilyMapper, get_default_torchsig_mapper


# Default impairment configurations
@dataclass
class ImpairmentConfig:
    """Configuration for signal impairments.

    Attributes:
        level: Impairment level (0=clean, 1=mild/cabled, 2=strong/wireless)
        snr_db_min: Minimum SNR in dB
        snr_db_max: Maximum SNR in dB
        cfo_normalized_std: Carrier frequency offset standard deviation (normalized)
        phase_noise_std: Phase noise standard deviation (radians)
        iq_imbalance_db: IQ amplitude imbalance in dB
        iq_phase_imbalance_deg: IQ phase imbalance in degrees
    """

    level: int = 1
    snr_db_min: float = -6.0
    snr_db_max: float = 20.0
    cfo_normalized_std: float = 0.01
    phase_noise_std: float = 0.01
    iq_imbalance_db: float = 1.0
    iq_phase_imbalance_deg: float = 5.0


# Default configurations for train and OOD splits
TRAIN_IMPAIRMENT_CONFIG = ImpairmentConfig(
    level=1,
    snr_db_min=-6.0,
    snr_db_max=20.0,
    cfo_normalized_std=0.01,
    phase_noise_std=0.01,
    iq_imbalance_db=1.0,
    iq_phase_imbalance_deg=5.0,
)

OOD_IMPAIRMENT_CONFIG = ImpairmentConfig(
    level=2,
    snr_db_min=-12.0,
    snr_db_max=6.0,
    cfo_normalized_std=0.05,
    phase_noise_std=0.05,
    iq_imbalance_db=3.0,
    iq_phase_imbalance_deg=15.0,
)


# Default modulations to generate (maps to TorchSig class names)
DEFAULT_MODULATIONS = [
    # PSK family
    "bpsk",
    "qpsk",
    "8psk",
    # FSK family
    "2fsk",
    "4fsk",
    "gfsk",
    "msk",
    # AM family
    "am-dsb",
    "am-dsb-sc",
    # SSB family
    "am-usb",
    "am-lsb",
    # QAM family
    "16qam",
    "64qam",
    "256qam",
]


class TorchSigDataset(Dataset):
    """PyTorch Dataset for TorchSig synthetic signals with family mapping.

    This dataset wraps TorchSig-generated signals or loads from cached numpy files.
    It supports random cropping from longer signals for input size flexibility.

    Args:
        data: Preloaded signal data with shape (N, signal_length) complex or (N, 2, signal_length)
        labels: Raw modulation labels (strings)
        snrs: SNR values for each sample
        family_mapper: FamilyMapper instance for label mapping
        transform: Optional transform to apply to samples
        crop_length: Length to crop signals to (default: 128)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        snrs: np.ndarray,
        family_mapper: Optional[FamilyMapper] = None,
        transform: Optional[Callable] = None,
        crop_length: int = 128,
        seed: int = 42,
    ):
        # Handle complex vs real I/Q format
        if np.iscomplexobj(data):
            # Convert complex to (N, 2, L) format
            self.data = np.stack([data.real, data.imag], axis=1).astype(np.float32)
        elif data.ndim == 3 and data.shape[1] == 2:
            # Already in (N, 2, L) format
            self.data = data.astype(np.float32)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        self.raw_labels = np.array(labels)
        self.snrs = snrs.astype(np.float32)
        self.family_mapper = family_mapper or get_default_torchsig_mapper()
        self.transform = transform
        self.crop_length = crop_length
        self.rng = np.random.default_rng(seed)

        # Map labels to family indices
        self.family_indices = np.array(
            [self.family_mapper.get_family_idx(str(lbl)) for lbl in self.raw_labels]
        )

        # Filter out unmapped labels
        valid_mask = self.family_indices != None  # noqa: E711
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"Warning: Filtering {n_invalid} samples with unmapped labels")
            self.data = self.data[valid_mask]
            self.raw_labels = self.raw_labels[valid_mask]
            self.snrs = self.snrs[valid_mask]
            self.family_indices = self.family_indices[valid_mask]

        self.family_indices = self.family_indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.family_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        """Get a sample with random cropping.

        Returns:
            x: Tensor of shape (2, crop_length)
            y_family: Integer family label
            meta: Dict with raw_label, snr, family_name, crop_start
        """
        x = self.data[idx]
        y_family = int(self.family_indices[idx])
        raw_label = str(self.raw_labels[idx])
        snr = float(self.snrs[idx])

        # Random crop
        signal_length = x.shape[1]
        if signal_length > self.crop_length:
            max_start = signal_length - self.crop_length
            start = self.rng.integers(0, max_start + 1)
            x = x[:, start : start + self.crop_length]
        else:
            start = 0
            # Pad if needed
            if signal_length < self.crop_length:
                pad_width = self.crop_length - signal_length
                x = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")

        # Apply transform
        if self.transform is not None:
            x = self.transform(x)

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x.copy())

        meta = {
            "raw_label": raw_label,
            "snr": snr,
            "family_name": self.family_mapper.family_names[y_family],
            "crop_start": start,
        }

        return x, y_family, meta

    @property
    def num_families(self) -> int:
        return self.family_mapper.num_families

    @property
    def family_names(self) -> list[str]:
        return self.family_mapper.family_names


def generate_torchsig_data(
    output_dir: str | Path,
    modulations: Optional[list[str]] = None,
    num_samples_per_class: int = 5000,
    signal_length: int = 1024,
    impairment_config: Optional[ImpairmentConfig] = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate TorchSig synthetic signal data.

    This function generates signals using TorchSig if available, otherwise
    falls back to simple synthetic generation for testing.

    Args:
        output_dir: Directory to save generated data
        modulations: List of modulation types to generate
        num_samples_per_class: Number of samples per modulation class
        signal_length: Length of each signal in samples
        impairment_config: Impairment configuration
        seed: Random seed for reproducibility

    Returns:
        data: Complex signal data with shape (N, signal_length)
        labels: Modulation labels (strings)
        snrs: SNR values
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modulations = modulations or DEFAULT_MODULATIONS
    config = impairment_config or TRAIN_IMPAIRMENT_CONFIG
    rng = np.random.default_rng(seed)

    try:
        # Try to use TorchSig
        from torchsig.datasets.modulations import ModulationsDataset
        from torchsig.transforms.target_transforms import DescToClassIndex

        print("Using TorchSig for signal generation...")

        # TorchSig generation
        all_data = []
        all_labels = []
        all_snrs = []

        for mod in modulations:
            print(f"  Generating {num_samples_per_class} samples of {mod}...")

            # Generate SNRs uniformly in range
            snrs = rng.uniform(
                config.snr_db_min, config.snr_db_max, size=num_samples_per_class
            )

            # Use TorchSig to generate signals
            dataset = ModulationsDataset(
                classes=[mod],
                use_class_idx=False,
                level=config.level,
                num_iq_samples=signal_length,
                num_samples=num_samples_per_class,
                include_snr=True,
            )

            for i, (signal, label, snr_val) in enumerate(dataset):
                if isinstance(signal, torch.Tensor):
                    signal = signal.numpy()
                all_data.append(signal)
                all_labels.append(mod)
                all_snrs.append(snrs[i])

        data = np.stack(all_data)
        labels = np.array(all_labels)
        snrs = np.array(all_snrs)

    except ImportError:
        print("TorchSig not available, using fallback synthetic generation...")
        data, labels, snrs = _generate_fallback_data(
            modulations, num_samples_per_class, signal_length, config, rng
        )

    # Save to disk
    np.save(output_dir / "data.npy", data)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "snrs.npy", snrs)

    metadata = {
        "modulations": modulations,
        "num_samples_per_class": num_samples_per_class,
        "signal_length": signal_length,
        "impairment_config": {
            "level": config.level,
            "snr_db_min": config.snr_db_min,
            "snr_db_max": config.snr_db_max,
        },
        "seed": seed,
        "total_samples": len(labels),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(labels)} samples, saved to {output_dir}")
    return data, labels, snrs


def _generate_fallback_data(
    modulations: list[str],
    num_samples_per_class: int,
    signal_length: int,
    config: ImpairmentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback synthetic signal generation when TorchSig is not available.

    Generates simple synthetic signals for testing and development.
    """
    all_data = []
    all_labels = []
    all_snrs = []

    for mod in modulations:
        print(f"  Generating {num_samples_per_class} samples of {mod} (fallback)...")
        snrs = rng.uniform(config.snr_db_min, config.snr_db_max, size=num_samples_per_class)

        for i in range(num_samples_per_class):
            signal = _generate_simple_signal(mod, signal_length, rng)

            # Add AWGN based on SNR
            snr_linear = 10 ** (snrs[i] / 10)
            signal_power = np.mean(np.abs(signal) ** 2)
            noise_power = signal_power / snr_linear
            noise = rng.normal(0, np.sqrt(noise_power / 2), signal_length) + 1j * rng.normal(
                0, np.sqrt(noise_power / 2), signal_length
            )
            signal = signal + noise

            # Normalize power
            signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))

            all_data.append(signal)
            all_labels.append(mod)
            all_snrs.append(snrs[i])

    data = np.stack(all_data)
    labels = np.array(all_labels)
    snrs = np.array(all_snrs)

    return data, labels, snrs


def _generate_simple_signal(mod: str, length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a simple signal for a given modulation type."""
    t = np.arange(length)
    mod_lower = mod.lower()

    if "bpsk" in mod_lower:
        symbols = rng.choice([-1, 1], size=length // 8)
        signal = np.repeat(symbols, 8)[:length].astype(np.complex128)
    elif "qpsk" in mod_lower:
        symbols = rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], size=length // 8) / np.sqrt(2)
        signal = np.repeat(symbols, 8)[:length]
    elif "8psk" in mod_lower:
        phases = np.exp(1j * 2 * np.pi * rng.integers(0, 8, size=length // 8) / 8)
        signal = np.repeat(phases, 8)[:length]
    elif "fsk" in mod_lower or "msk" in mod_lower:
        freq_idx = rng.integers(0, 2, size=length // 16)
        freqs = np.repeat(freq_idx, 16)[:length]
        signal = np.exp(1j * 2 * np.pi * 0.1 * (2 * freqs - 1) * t / length)
    elif "am" in mod_lower or "dsb" in mod_lower:
        carrier = np.exp(1j * 2 * np.pi * 0.25 * t)
        message = np.sin(2 * np.pi * 0.01 * t)
        signal = (1 + 0.5 * message) * carrier
    elif "ssb" in mod_lower or "usb" in mod_lower or "lsb" in mod_lower:
        carrier = np.exp(1j * 2 * np.pi * 0.25 * t)
        message = np.sin(2 * np.pi * 0.01 * t)
        signal = message * carrier
    elif "qam" in mod_lower:
        # Simple 16-QAM
        constellation = np.array(
            [a + 1j * b for a in [-3, -1, 1, 3] for b in [-3, -1, 1, 3]]
        ) / np.sqrt(10)
        symbols = rng.choice(constellation, size=length // 8)
        signal = np.repeat(symbols, 8)[:length]
    else:
        # Default: random complex signal
        signal = rng.normal(0, 1, length) + 1j * rng.normal(0, 1, length)

    return signal.astype(np.complex64)


def load_torchsig_data(
    cache_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load TorchSig data from cache directory.

    Args:
        cache_dir: Directory containing cached data files

    Returns:
        data: Signal data
        labels: Modulation labels
        snrs: SNR values
    """
    cache_dir = Path(cache_dir)

    data = np.load(cache_dir / "data.npy")
    labels = np.load(cache_dir / "labels.npy", allow_pickle=True)
    snrs = np.load(cache_dir / "snrs.npy")

    return data, labels, snrs


def get_torchsig_loaders(
    cache_dir: str | Path,
    batch_size: int = 256,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    family_mapper: Optional[FamilyMapper] = None,
    crop_length: int = 128,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    generate_if_missing: bool = True,
    generation_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Create DataLoaders for TorchSig dataset.

    Args:
        cache_dir: Directory containing cached data
        batch_size: Batch size for all loaders
        train_transform: Transform for training data
        eval_transform: Transform for validation and test data
        family_mapper: FamilyMapper instance (default: TorchSig mapper)
        crop_length: Length to crop signals to
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        generate_if_missing: Whether to generate data if not cached
        generation_config: Config for data generation if needed

    Returns:
        Dict with "train", "val", "test" DataLoaders and "family_names" list
    """
    cache_dir = Path(cache_dir)
    family_mapper = family_mapper or get_default_torchsig_mapper()

    # Check if data exists
    if not (cache_dir / "data.npy").exists():
        if generate_if_missing:
            gen_config = generation_config or {}
            generate_torchsig_data(cache_dir, seed=seed, **gen_config)
        else:
            raise FileNotFoundError(
                f"TorchSig cache not found at {cache_dir}. "
                "Run data generation first or set generate_if_missing=True."
            )

    # Load data
    data, labels, snrs = load_torchsig_data(cache_dir)

    # Create stratified splits
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Get family indices for stratification
    family_indices = np.array(
        [family_mapper.get_family_idx(str(lbl)) or -1 for lbl in labels]
    )
    valid_mask = family_indices >= 0
    data = data[valid_mask]
    labels = labels[valid_mask]
    snrs = snrs[valid_mask]
    family_indices = family_indices[valid_mask]

    # Stratify by (family, snr_bin)
    snr_bins = np.round(snrs / 2) * 2  # 2 dB bins
    strat_key = family_indices * 1000 + snr_bins.astype(int)

    # Split
    idx_train, idx_temp = train_test_split(
        np.arange(len(labels)),
        train_size=train_ratio,
        stratify=strat_key,
        random_state=seed,
    )

    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    strat_key_temp = strat_key[idx_temp]

    idx_val, idx_test = train_test_split(
        idx_temp,
        train_size=val_test_ratio,
        stratify=strat_key_temp,
        random_state=seed,
    )

    # Create datasets
    train_dataset = TorchSigDataset(
        data[idx_train],
        labels[idx_train],
        snrs[idx_train],
        family_mapper=family_mapper,
        transform=train_transform,
        crop_length=crop_length,
        seed=seed,
    )
    val_dataset = TorchSigDataset(
        data[idx_val],
        labels[idx_val],
        snrs[idx_val],
        family_mapper=family_mapper,
        transform=eval_transform,
        crop_length=crop_length,
        seed=seed + 1,
    )
    test_dataset = TorchSigDataset(
        data[idx_test],
        labels[idx_test],
        snrs[idx_test],
        family_mapper=family_mapper,
        transform=eval_transform,
        crop_length=crop_length,
        seed=seed + 2,
    )

    # Create collate function for compatibility with existing trainers
    def family_collate_fn(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        snr = torch.tensor([item[2]["snr"] for item in batch], dtype=torch.float32)
        return x, y, snr

    # Create loaders
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=family_collate_fn,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=family_collate_fn,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=family_collate_fn,
        ),
        "family_names": family_mapper.family_names,
    }

    return loaders