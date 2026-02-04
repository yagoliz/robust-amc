"""RadioML2018.01a dataset loader with cross-dataset support."""

from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# RadioML2018.01a has 24 modulation classes
MODULATION_CLASSES_2018 = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]

# SNR levels in RadioML2018.01a: -20 to +30 dB in 2 dB steps
SNR_LEVELS_2018 = list(range(-20, 32, 2))

# Mapping from 2018 class names to 2016 class names (for overlapping classes)
# Only includes classes that have clear equivalents in both datasets
CLASS_NAME_MAPPING_2018_TO_2016 = {
    "BPSK": "BPSK",
    "QPSK": "QPSK",
    "8PSK": "8PSK",
    "16QAM": "QAM16",
    "64QAM": "QAM64",
    "GMSK": "GFSK",      # GMSK ≈ GFSK (similar Gaussian filtering)
    "FM": "WBFM",        # FM ≈ WBFM (wideband FM)
    "AM-DSB-SC": "AM-DSB",  # AM-DSB suppressed carrier variant
}

# Reverse mapping
CLASS_NAME_MAPPING_2016_TO_2018 = {v: k for k, v in CLASS_NAME_MAPPING_2018_TO_2016.items()}

# Classes that exist in both datasets (using 2016 naming convention)
# Note: CPFSK, PAM4, AM-SSB don't have equivalents in 2018
OVERLAPPING_CLASSES = list(CLASS_NAME_MAPPING_2018_TO_2016.values())


def load_radioml2018a(
    data_path: str | Path,
    split_segments: bool = True,
    segment_length: int = 128,
    overlapping_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load RadioML2018.01a dataset from HDF5 file.

    Args:
        data_path: Path to 2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 file
        split_segments: If True, split 1024-sample signals into 128-sample segments
        segment_length: Length of each segment (default 128 to match 2016)
        overlapping_only: If True, only load classes that overlap with 2016

    Returns:
        data: I/Q samples with shape (N, 2, seq_len)
        labels: Class indices
        snrs: SNR values for each sample
        class_names: List of class names used
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
        )

    with h5py.File(data_path, "r") as f:
        # RadioML2018.01a structure: X (samples), Y (one-hot labels), Z (SNR)
        X = f["X"][:]  # Shape: (N, 1024, 2)
        Y = f["Y"][:]  # Shape: (N, 24) one-hot encoded
        Z = f["Z"][:]  # Shape: (N,) SNR values

    # Convert one-hot to class indices
    labels = np.argmax(Y, axis=1)
    snrs = Z.flatten()

    # Transpose to match 2016 format: (N, 2, 1024)
    data = X.transpose(0, 2, 1).astype(np.float32)

    # Determine which classes to use
    if overlapping_only:
        # Filter to only overlapping classes
        class_names = list(CLASS_NAME_MAPPING_2018_TO_2016.keys())
        class_indices_to_keep = [MODULATION_CLASSES_2018.index(c) for c in class_names if c in MODULATION_CLASSES_2018]  # noqa: E501

        # Create mask for samples with overlapping classes
        mask = np.isin(labels, class_indices_to_keep)
        data = data[mask]
        labels = labels[mask]
        snrs = snrs[mask]

        # Remap labels to consecutive indices
        old_to_new = {old: new for new, old in enumerate(sorted(class_indices_to_keep))}
        labels = np.array([old_to_new[label] for label in labels])

        # Update class names to match new indices
        class_names = [MODULATION_CLASSES_2018[i] for i in sorted(class_indices_to_keep)]
    else:
        class_names = MODULATION_CLASSES_2018.copy()

    # Split into segments if requested
    if split_segments and data.shape[2] > segment_length:
        data, labels, snrs = _split_into_segments(data, labels, snrs, segment_length)

    return data, labels, snrs, class_names


def _split_into_segments(
    data: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    segment_length: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split long samples into multiple shorter segments.

    Args:
        data: Shape (N, 2, seq_len) where seq_len > segment_length
        labels: Shape (N,)
        snrs: Shape (N,)
        segment_length: Target segment length

    Returns:
        data: Shape (N * n_segments, 2, segment_length)
        labels: Shape (N * n_segments,) - repeated for each segment
        snrs: Shape (N * n_segments,) - repeated for each segment
    """
    seq_len = data.shape[2]
    n_segments = seq_len // segment_length

    if n_segments <= 1:
        return data[:, :, :segment_length], labels, snrs

    # Reshape: (N, 2, 1024) -> (N, 2, 8, 128) -> (N, 8, 2, 128) -> (N*8, 2, 128)
    n_samples = data.shape[0]
    segments = data[:, :, :n_segments * segment_length]  # Trim to exact multiple
    segments = segments.reshape(n_samples, 2, n_segments, segment_length)
    segments = segments.transpose(0, 2, 1, 3)  # (N, n_segments, 2, segment_length)
    segments = segments.reshape(-1, 2, segment_length)  # (N * n_segments, 2, segment_length)

    # Repeat labels and SNRs for each segment
    labels = np.repeat(labels, n_segments)
    snrs = np.repeat(snrs, n_segments)

    return segments, labels, snrs


class RadioML2018Dataset(Dataset):
    """PyTorch Dataset for RadioML2018.01a.

    Args:
        data: I/Q samples with shape (N, 2, seq_len)
        labels: Class indices
        snrs: SNR values for each sample
        class_names: List of class names
        transform: Optional transform to apply to samples
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        snrs: np.ndarray,
        class_names: list[str],
        transform: Optional[Callable] = None,
    ):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.snrs = snrs.astype(np.float32)
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float]:
        x = self.data[idx]
        y = self.labels[idx]
        snr = self.snrs[idx]

        if self.transform is not None:
            x = self.transform(x)

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        return x, y, snr

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def stratified_split_2018(
    data: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    max_samples_per_split: Optional[int] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create stratified train/val/test splits for RadioML2018.

    Args:
        data: I/Q samples
        labels: Class indices
        snrs: SNR values
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_state: Random seed
        max_samples_per_split: Optional limit on samples per split (for memory)

    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Create stratification key from (label, snr) pairs
    # Bin SNRs to handle the continuous range
    snr_bins = np.round(snrs / 2) * 2  # Round to nearest 2 dB
    strat_key = labels * 1000 + snr_bins.astype(int)

    # First split: train vs (val + test)
    idx_train, idx_temp = train_test_split(
        np.arange(len(labels)),
        train_size=train_ratio,
        stratify=strat_key,
        random_state=random_state,
    )

    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    strat_key_temp = strat_key[idx_temp]

    idx_val, idx_test = train_test_split(
        idx_temp,
        train_size=val_test_ratio,
        stratify=strat_key_temp,
        random_state=random_state,
    )

    # Optionally limit samples
    if max_samples_per_split is not None:
        if len(idx_train) > max_samples_per_split:
            idx_train = np.random.choice(idx_train, max_samples_per_split, replace=False)
        if len(idx_val) > max_samples_per_split:
            idx_val = np.random.choice(idx_val, max_samples_per_split, replace=False)
        if len(idx_test) > max_samples_per_split:
            idx_test = np.random.choice(idx_test, max_samples_per_split, replace=False)

    return {
        "train": (data[idx_train], labels[idx_train], snrs[idx_train]),
        "val": (data[idx_val], labels[idx_val], snrs[idx_val]),
        "test": (data[idx_test], labels[idx_test], snrs[idx_test]),
    }


def get_data_loaders_2018(
    data_path: str | Path,
    batch_size: int = 256,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    num_workers: int = 4,
    random_state: int = 42,
    split_segments: bool = True,
    overlapping_only: bool = False,
    max_samples: Optional[int] = None,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders for RadioML2018.01a.

    Args:
        data_path: Path to HDF5 file
        batch_size: Batch size for all loaders
        train_transform: Transform for training data
        eval_transform: Transform for validation and test data
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        num_workers: Number of data loading workers
        random_state: Random seed
        split_segments: Whether to split 1024 samples into 128-sample segments
        overlapping_only: Whether to only use classes that overlap with 2016
        max_samples: Optional limit on total samples (useful for testing)

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders and 'class_names'
    """
    # Load raw data
    data, labels, snrs, class_names = load_radioml2018a(
        data_path,
        split_segments=split_segments,
        overlapping_only=overlapping_only,
    )

    # Optionally limit samples
    if max_samples is not None and len(data) > max_samples:
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data[indices]
        labels = labels[indices]
        snrs = snrs[indices]

    # Create stratified splits
    splits = stratified_split_2018(
        data, labels, snrs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # Create datasets
    train_dataset = RadioML2018Dataset(
        *splits["train"], class_names=class_names, transform=train_transform
    )
    val_dataset = RadioML2018Dataset(
        *splits["val"], class_names=class_names, transform=eval_transform
    )
    test_dataset = RadioML2018Dataset(
        *splits["test"], class_names=class_names, transform=eval_transform
    )

    # Create data loaders
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "class_names": class_names,
    }

    return loaders
