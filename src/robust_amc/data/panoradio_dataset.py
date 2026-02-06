"""Panoradio HF radio signal dataset loader.

This module provides a PyTorch Dataset for the Panoradio Radio Signal
Classification Dataset containing real HF radio signals with:
- 18 HF transmission modes
- Watterson fading channel model
- Random frequency offset (±250 Hz)
- SNR range: -10 to +25 dB

Dataset source: https://panoradio-sdr.de/radio-signal-classification-dataset/
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .label_mapping import FamilyMapper, get_default_panoradio_mapper


# Panoradio dataset constants
PANORADIO_SAMPLE_RATE = 6000  # 6 kHz
PANORADIO_SIGNAL_LENGTH = 2048
PANORADIO_SNR_LEVELS = [25, 20, 15, 10, 5, 0, -5, -10]


class PanoradioDataset(Dataset):
    """PyTorch Dataset for Panoradio HF radio signals.

    This dataset loads real HF radio captures with random windowing to
    extract shorter segments for model input.

    Args:
        data: Signal data with shape (N, signal_length) complex or (N, 2, signal_length)
        labels: Mode labels (strings)
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
        self.family_mapper = family_mapper or get_default_panoradio_mapper()
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

    @property
    def available_snrs(self) -> list[int]:
        return PANORADIO_SNR_LEVELS


def load_panoradio_metadata(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Panoradio metadata from tags.csv.

    The tags.csv file contains the label and SNR for each sample.
    Expected format: one row per sample with columns for label and SNR.

    Args:
        data_dir: Directory containing tags.csv

    Returns:
        labels: Array of mode labels (strings)
        snrs: Array of SNR values
    """
    data_dir = Path(data_dir)
    tags_path = data_dir / "tags.csv"

    if not tags_path.exists():
        raise FileNotFoundError(f"tags.csv not found at {tags_path}")

    labels = []
    snrs = []

    with open(tags_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Determine column indices
        if header:
            # Try to find columns by name
            header_lower = [h.lower().strip() for h in header]
            try:
                label_idx = header_lower.index("label") if "label" in header_lower else 0
                snr_idx = header_lower.index("snr") if "snr" in header_lower else 1
            except ValueError:
                # Fallback to positional
                label_idx = 0
                snr_idx = 1
        else:
            label_idx = 0
            snr_idx = 1

        for row in reader:
            if len(row) >= 2:
                labels.append(row[label_idx].strip())
                try:
                    snrs.append(float(row[snr_idx]))
                except ValueError:
                    snrs.append(0.0)

    return np.array(labels), np.array(snrs, dtype=np.float32)


def load_panoradio_data(
    data_dir: str | Path,
    mmap_mode: Optional[str] = "r",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Panoradio dataset from directory.

    Expected files:
    - rscd_2048.npy: Signal data (172800, 2048) complex
    - tags.csv: Labels and SNR metadata

    Args:
        data_dir: Directory containing Panoradio data files
        mmap_mode: Memory-map mode for large files (default: "r" for read-only)

    Returns:
        data: Signal data with shape (N, 2048) complex
        labels: Mode labels (strings)
        snrs: SNR values
    """
    data_dir = Path(data_dir)

    # Look for data file
    data_path = None
    for candidate in ["rscd_2048.npy", "data.npy", "signals.npy"]:
        if (data_dir / candidate).exists():
            data_path = data_dir / candidate
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Signal data not found in {data_dir}. "
            "Expected rscd_2048.npy, data.npy, or signals.npy"
        )

    # Load data with memory mapping for large files
    data = np.load(data_path, mmap_mode=mmap_mode)
    print(f"Loaded Panoradio data: {data.shape}, dtype={data.dtype}")

    # Load metadata
    labels, snrs = load_panoradio_metadata(data_dir)

    # Verify alignment
    if len(labels) != len(data):
        raise ValueError(
            f"Mismatch: {len(data)} signals but {len(labels)} labels in metadata"
        )

    return data, labels, snrs


def get_panoradio_loaders(
    data_dir: str | Path,
    batch_size: int = 256,
    transform: Optional[Callable] = None,
    family_mapper: Optional[FamilyMapper] = None,
    crop_length: int = 128,
    train_ratio: float = 0.0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.8,
    num_workers: int = 4,
    seed: int = 42,
    snr_filter: Optional[tuple[float, float]] = None,
    include_unmapped: bool = False,
    device: str = "cpu"
) -> dict[str, Any]:
    """Create DataLoaders for Panoradio dataset.

    By default, Panoradio is used for zero-shot evaluation with no training split.
    Set train_ratio > 0 for few-shot fine-tuning experiments.

    Args:
        data_dir: Directory containing Panoradio data
        batch_size: Batch size for all loaders
        transform: Transform to apply to all splits
        family_mapper: FamilyMapper instance (default: Panoradio mapper)
        crop_length: Length to crop signals to
        train_ratio: Fraction for training (default 0 for zero-shot)
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        snr_filter: Optional (min, max) SNR filter
        include_unmapped: Whether to include samples that don't map to families
        device: Device where it will be loaded (to avoid issues with pin_memory)

    Returns:
        Dict with DataLoaders and "family_names" list.
        If train_ratio > 0, includes "train" loader.
    """
    data_dir = Path(data_dir)
    family_mapper = family_mapper or get_default_panoradio_mapper()

    # Load data
    data, labels, snrs = load_panoradio_data(data_dir)

    # Optional SNR filtering
    if snr_filter is not None:
        snr_min, snr_max = snr_filter
        snr_mask = (snrs >= snr_min) & (snrs <= snr_max)
        data = data[snr_mask]
        labels = labels[snr_mask]
        snrs = snrs[snr_mask]
        print(f"SNR filter [{snr_min}, {snr_max}]: {snr_mask.sum()} samples retained")

    # Filter to mapped labels if requested
    if not include_unmapped:
        mapped_mask = np.array([family_mapper.is_mapped(str(lbl)) for lbl in labels])
        data = data[mapped_mask]
        labels = labels[mapped_mask]
        snrs = snrs[mapped_mask]
        print(f"Family mapping filter: {mapped_mask.sum()} samples retained")

    # Get family indices for stratification
    family_indices = np.array(
        [family_mapper.get_family_idx(str(lbl)) or -1 for lbl in labels]
    )

    # Stratify by (family, snr)
    strat_key = family_indices * 1000 + snrs.astype(int)

    # Create collate function for compatibility
    def family_collate_fn(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        snr = torch.tensor([item[2]["snr"] for item in batch], dtype=torch.float32)
        return x, y, snr

    loaders: dict[str, Any] = {"family_names": family_mapper.family_names}

    # Warning suppression: We need to set pin_memory to false on MPS
    pin_memory = False if device == "mps" else True

    if train_ratio > 0:
        # Three-way split with training
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

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

        # Create training dataset
        train_dataset = PanoradioDataset(
            data[idx_train],
            labels[idx_train],
            snrs[idx_train],
            family_mapper=family_mapper,
            transform=transform,
            crop_length=crop_length,
            seed=seed,
        )

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=family_collate_fn,
        )
    else:
        # Two-way split (val/test only) for zero-shot evaluation
        assert abs(val_ratio + test_ratio - 1.0) < 1e-6

        idx_val, idx_test = train_test_split(
            np.arange(len(labels)),
            train_size=val_ratio,
            stratify=strat_key,
            random_state=seed,
        )

    # Create val/test datasets
    val_dataset = PanoradioDataset(
        data[idx_val],
        labels[idx_val],
        snrs[idx_val],
        family_mapper=family_mapper,
        transform=transform,
        crop_length=crop_length,
        seed=seed + 1,
    )

    test_dataset = PanoradioDataset(
        data[idx_test],
        labels[idx_test],
        snrs[idx_test],
        family_mapper=family_mapper,
        transform=transform,
        crop_length=crop_length,
        seed=seed + 2,
    )

    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=family_collate_fn,
    )

    loaders["test"] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=family_collate_fn,
    )

    return loaders


def get_panoradio_by_snr(
    data_dir: str | Path,
    snr: int,
    family_mapper: Optional[FamilyMapper] = None,
    transform: Optional[Callable] = None,
    crop_length: int = 128,
    seed: int = 42,
) -> PanoradioDataset:
    """Get Panoradio dataset filtered to a specific SNR level.

    Useful for SNR-specific evaluation.

    Args:
        data_dir: Directory containing Panoradio data
        snr: Target SNR level in dB
        family_mapper: FamilyMapper instance
        transform: Transform to apply
        crop_length: Length to crop signals to
        seed: Random seed

    Returns:
        PanoradioDataset containing only samples at the specified SNR
    """
    data, labels, snrs = load_panoradio_data(data_dir)

    # Filter by SNR
    snr_mask = snrs == snr
    data = data[snr_mask]
    labels = labels[snr_mask]
    snrs = snrs[snr_mask]

    return PanoradioDataset(
        data,
        labels,
        snrs,
        family_mapper=family_mapper or get_default_panoradio_mapper(),
        transform=transform,
        crop_length=crop_length,
        seed=seed,
    )
