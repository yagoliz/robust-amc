"""Contrastive data pipeline for CLSR-AMC training.

This module provides dataset wrappers and loaders that produce 5-tuple batches
(x_i, x_j, x_orig, y, meta) required by CLSRAMCTrainer, where:
- x_i, x_j: Two independently augmented views of the same signal
- x_orig: Clean (power-normalized) version for reconstruction
- y: Family label
- meta: Metadata dict with raw_label, snr, family_name
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .augmentations import MDADMCPipeline
from .label_mapping import FamilyMapper
from .torchsig_dataset import (
    TorchSigDataset,
    _load_and_split_torchsig,
    family_collate_fn,
)
from .transforms import Compose, PowerNormalize, ToTensor


class ContrastiveDataset(Dataset):
    """Wraps a TorchSigDataset to produce contrastive 5-tuple samples.

    For each sample, applies augmentation independently twice to create two
    views (x_i, x_j), and applies a clean transform for x_orig.

    Args:
        base_dataset: A TorchSigDataset with transform=None.
        augmentation: Stochastic augmentation (e.g. MDADMCPipeline).
        clean_transform: Deterministic transform for x_orig (e.g. PowerNormalize+ToTensor).
    """

    def __init__(
        self,
        base_dataset: TorchSigDataset,
        augmentation: Callable,
        clean_transform: Callable,
    ):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.clean_transform = clean_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict]:
        """Get a contrastive sample.

        Returns:
            (x_i, x_j, x_orig, y_family, meta) where x_i and x_j are
            independently augmented views and x_orig is the clean signal.
        """
        # Base dataset returns (x_numpy, y_family, meta) with transform=None
        x_raw, y_family, meta = self.base_dataset[idx]

        # x_raw is a tensor from TorchSigDataset.__getitem__ (numpy->tensor conversion)
        # Convert back to numpy for augmentation transforms that expect numpy
        if isinstance(x_raw, torch.Tensor):
            x_np = x_raw.numpy()
        else:
            x_np = x_raw

        # Two independent augmented views
        x_i = self.augmentation(x_np.copy())
        x_j = self.augmentation(x_np.copy())

        # Clean version for reconstruction target
        x_orig = self.clean_transform(x_np.copy())

        # Ensure augmented views go through clean_transform too (normalize + to_tensor)
        x_i = self.clean_transform(x_i)
        x_j = self.clean_transform(x_j)

        return x_i, x_j, x_orig, y_family, meta


def contrastive_collate_fn(batch):
    """Collate function for contrastive 5-tuple batches.

    Stacks x_i, x_j, x_orig into batch tensors, y into a label tensor,
    and collects SNR from meta dicts.
    """
    x_i = torch.stack([item[0] for item in batch])
    x_j = torch.stack([item[1] for item in batch])
    x_orig = torch.stack([item[2] for item in batch])
    y = torch.tensor([item[3] for item in batch], dtype=torch.long)
    meta = {"snr": torch.tensor([item[4]["snr"] for item in batch], dtype=torch.float32)}
    return x_i, x_j, x_orig, y, meta


def get_contrastive_loaders(
    cache_dir: str,
    batch_size: int = 256,
    augmentation: Optional[Callable] = None,
    clean_transform: Optional[Callable] = None,
    family_mapper: Optional[FamilyMapper] = None,
    crop_length: int = 128,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    generate_if_missing: bool = True,
    generation_config: Optional[dict] = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Create contrastive DataLoaders for CLSR-AMC training.

    Train and val loaders produce 5-tuple batches (x_i, x_j, x_orig, y, meta).
    The test loader produces standard 3-tuple batches (x, y, meta) for evaluation.

    Args:
        cache_dir: Directory containing cached TorchSig data
        batch_size: Batch size
        augmentation: Stochastic augmentation for contrastive views.
            Defaults to MDADMCPipeline(p=0.5).
        clean_transform: Transform for x_orig and final normalization.
            Defaults to PowerNormalize() + ToTensor().
        family_mapper: FamilyMapper instance
        crop_length: Length to crop signals to
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        num_workers: DataLoader workers
        seed: Random seed
        generate_if_missing: Generate data if cache is missing
        generation_config: Config for data generation
        device: Target device (affects pin_memory)

    Returns:
        Dict with "train" (5-tuple), "val" (5-tuple), "test" (3-tuple)
        DataLoaders and "family_names" list.
    """
    if augmentation is None:
        augmentation = MDADMCPipeline(p=0.5)
    if clean_transform is None:
        clean_transform = Compose([PowerNormalize(), ToTensor()])

    # Load and split with transform=None (raw numpy signals)
    train_ds, val_ds, test_ds, family_mapper = _load_and_split_torchsig(
        cache_dir=cache_dir,
        family_mapper=family_mapper,
        transform=None,
        crop_length=crop_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        generate_if_missing=generate_if_missing,
        generation_config=generation_config,
    )

    # Wrap train/val in ContrastiveDataset
    train_contrastive = ContrastiveDataset(train_ds, augmentation, clean_transform)
    val_contrastive = ContrastiveDataset(val_ds, augmentation, clean_transform)

    # Test uses standard pipeline (evaluation doesn't need contrastive pairs)
    test_ds.transform = clean_transform

    pin_memory = False if device == "mps" else True

    loaders = {
        "train": DataLoader(
            train_contrastive,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=contrastive_collate_fn,
        ),
        "val": DataLoader(
            val_contrastive,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=contrastive_collate_fn,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=family_collate_fn,
        ),
        "family_names": family_mapper.family_names,
    }

    return loaders