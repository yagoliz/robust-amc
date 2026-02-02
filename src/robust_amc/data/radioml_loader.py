"""RadioML2016.10a dataset loader with stratified splits."""

import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# RadioML2016.10a modulation classes
MODULATION_CLASSES = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]

# SNR levels in the dataset
SNR_LEVELS = list(range(-20, 20, 2))  # -20 to 18 dB in 2 dB steps


class RadioMLDataset(Dataset):
    """PyTorch Dataset for RadioML2016.10a.

    Args:
        data: I/Q samples with shape (N, 2, 128)
        labels: Modulation class indices
        snrs: SNR values for each sample
        transform: Optional transform to apply to samples
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        snrs: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.snrs = snrs.astype(np.float32)
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
        return len(MODULATION_CLASSES)

    @property
    def class_names(self) -> list[str]:
        return MODULATION_CLASSES


def load_radioml2016a(
    data_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load RadioML2016.10a dataset from pickle file.

    The dataset should be downloaded from:
    https://www.deepsig.ai/datasets or via torchsig library.

    Args:
        data_path: Path to RML2016.10a_dict.pkl file

    Returns:
        data: I/Q samples with shape (N, 2, 128)
        labels: Modulation class indices
        snrs: SNR values for each sample
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please download RadioML2016.10a from https://www.deepsig.ai/datasets"
        )

    with open(data_path, "rb") as f:
        raw_data = pickle.load(f, encoding="latin1")

    # Build label to index mapping
    mod_to_idx = {mod: idx for idx, mod in enumerate(MODULATION_CLASSES)}

    # Extract data, labels, and SNRs
    data_list = []
    label_list = []
    snr_list = []

    for (mod, snr), samples in raw_data.items():
        if mod not in mod_to_idx:
            continue  # Skip unknown modulations

        n_samples = samples.shape[0]
        data_list.append(samples)
        label_list.extend([mod_to_idx[mod]] * n_samples)
        snr_list.extend([snr] * n_samples)

    data = np.vstack(data_list)
    labels = np.array(label_list)
    snrs = np.array(snr_list)

    return data, labels, snrs


def stratified_split(
    data: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create stratified train/val/test splits.

    Stratification is done by (modulation, SNR) to ensure balanced
    distribution across all conditions.

    Args:
        data: I/Q samples
        labels: Modulation class indices
        snrs: SNR values
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        (data, labels, snrs) tuple
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Create stratification key from (label, snr) pairs
    strat_key = labels * 1000 + snrs.astype(int)  # Unique key per (mod, snr)

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

    return {
        "train": (data[idx_train], labels[idx_train], snrs[idx_train]),
        "val": (data[idx_val], labels[idx_val], snrs[idx_val]),
        "test": (data[idx_test], labels[idx_test], snrs[idx_test]),
    }


def get_data_loaders(
    data_path: str | Path,
    batch_size: int = 256,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    num_workers: int = 4,
    random_state: int = 42,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders for RadioML2016.10a.

    Args:
        data_path: Path to dataset pickle file
        batch_size: Batch size for all loaders
        train_transform: Transform for training data
        eval_transform: Transform for validation and test data
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        num_workers: Number of data loading workers
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Load raw data
    data, labels, snrs = load_radioml2016a(data_path)

    # Create stratified splits
    splits = stratified_split(
        data, labels, snrs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # Create datasets
    train_dataset = RadioMLDataset(*splits["train"], transform=train_transform)
    val_dataset = RadioMLDataset(*splits["val"], transform=eval_transform)
    test_dataset = RadioMLDataset(*splits["test"], transform=eval_transform)

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
    }

    return loaders


def get_samples_by_modulation_snr(
    dataset: RadioMLDataset,
    modulation: str,
    snr: int,
    n_samples: int = 10,
) -> tuple[torch.Tensor, int]:
    """Get samples for a specific modulation and SNR.

    Useful for visualization and debugging.

    Args:
        dataset: RadioML dataset
        modulation: Modulation name (e.g., "QPSK")
        snr: SNR value in dB
        n_samples: Number of samples to return

    Returns:
        samples: Tensor of shape (n_samples, 2, 128)
        label: Modulation class index
    """
    mod_idx = MODULATION_CLASSES.index(modulation)

    # Find matching indices
    mask = (dataset.labels == mod_idx) & (dataset.snrs == snr)
    indices = np.where(mask)[0]

    if len(indices) < n_samples:
        raise ValueError(
            f"Only {len(indices)} samples available for {modulation} at {snr}dB"
        )

    # Sample randomly
    selected = np.random.choice(indices, size=n_samples, replace=False)
    samples = torch.from_numpy(dataset.data[selected])

    return samples, mod_idx
