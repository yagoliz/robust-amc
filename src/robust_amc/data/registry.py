"""Dataset registry for unified dataset loading and configuration.

This module provides a centralized interface for loading datasets from
YAML configuration files, supporting TorchSig and Panoradio datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from torch.utils.data import DataLoader


class DatasetType(Enum):
    """Supported dataset types."""

    TORCHSIG = "torchsig"
    PANORADIO = "panoradio"


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    Attributes:
        dataset_type: Type of dataset (torchsig, panoradio)
        data_path: Path to data directory or file
        label_map_path: Path to family mapping YAML
        batch_size: Batch size for DataLoaders
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        extra_config: Additional dataset-specific configuration
    """

    dataset_type: DatasetType
    data_path: str | Path
    label_map_path: Optional[str | Path] = None
    batch_size: int = 256
    num_workers: int = 4
    seed: int = 42
    extra_config: dict = field(default_factory=dict)


def load_config_from_yaml(yaml_path: str | Path) -> DatasetConfig:
    """Load dataset configuration from a YAML file.

    Example YAML structure:
        dataset_type: torchsig
        data_path: data/torchsig_cache
        label_map_path: configs/label_maps/torchsig_to_family.yaml
        batch_size: 256
        num_workers: 4
        seed: 42
        extra_config:
          crop_length: 128
          impairments:
            level: 1
            snr_db_min: -6
            snr_db_max: 20

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        DatasetConfig instance
    """
    yaml_path = Path(yaml_path)

    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)

    # Parse dataset type
    dataset_type_str = config_dict.get("dataset_type", "").lower()
    try:
        dataset_type = DatasetType(dataset_type_str)
    except ValueError:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type_str}. "
            f"Supported: {[t.value for t in DatasetType]}"
        )

    return DatasetConfig(
        dataset_type=dataset_type,
        data_path=config_dict.get("data_path", ""),
        label_map_path=config_dict.get("label_map_path"),
        batch_size=config_dict.get("batch_size", 256),
        num_workers=config_dict.get("num_workers", 4),
        seed=config_dict.get("seed", 42),
        extra_config=config_dict.get("extra_config", {}),
    )


def get_loaders(
    config: DatasetConfig,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    device: str = "cpu"
) -> dict[str, Any]:
    """Get DataLoaders from a dataset configuration.

    Args:
        config: Dataset configuration
        train_transform: Transform for training data
        eval_transform: Transform for evaluation data
        device: Where the data will be loaded

    Returns:
        Dict with DataLoaders ("train", "val", "test") and metadata
    """
    from .label_mapping import FamilyMapper

    # Load family mapper if specified
    family_mapper = None
    if config.label_map_path:
        family_mapper = FamilyMapper(config.label_map_path)

    if config.dataset_type == DatasetType.TORCHSIG:
        from .torchsig_dataset import get_torchsig_loaders

        extra = config.extra_config
        return get_torchsig_loaders(
            cache_dir=config.data_path,
            batch_size=config.batch_size,
            train_transform=train_transform,
            eval_transform=eval_transform,
            family_mapper=family_mapper,
            crop_length=extra.get("crop_length", 128),
            train_ratio=extra.get("train_ratio", 0.6),
            val_ratio=extra.get("val_ratio", 0.2),
            test_ratio=extra.get("test_ratio", 0.2),
            num_workers=config.num_workers,
            seed=config.seed,
            generate_if_missing=extra.get("generate_if_missing", True),
            generation_config=extra.get("generation_config"),
            device=device
        )

    elif config.dataset_type == DatasetType.PANORADIO:
        from .panoradio_dataset import get_panoradio_loaders

        extra = config.extra_config
        return get_panoradio_loaders(
            data_dir=config.data_path,
            batch_size=config.batch_size,
            transform=eval_transform,  # Panoradio typically uses same transform
            family_mapper=family_mapper,
            crop_length=extra.get("crop_length", 128),
            train_ratio=extra.get("train_ratio", 0.0),
            val_ratio=extra.get("val_ratio", 0.2),
            test_ratio=extra.get("test_ratio", 0.8),
            num_workers=config.num_workers,
            seed=config.seed,
            snr_filter=extra.get("snr_filter"),
            include_unmapped=extra.get("include_unmapped", False),
            device=device
        )

    else:
        raise ValueError(f"Unsupported dataset type: {config.dataset_type}")


def get_loaders_from_yaml(
    yaml_path: str | Path,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
) -> dict[str, Any]:
    """Convenience function to load DataLoaders directly from YAML config.

    Args:
        yaml_path: Path to YAML configuration file
        train_transform: Transform for training data
        eval_transform: Transform for evaluation data

    Returns:
        Dict with DataLoaders and metadata
    """
    config = load_config_from_yaml(yaml_path)
    return get_loaders(config, train_transform, eval_transform)


# Registry of dataset factory functions (for programmatic access)
DATASET_REGISTRY = {
    DatasetType.TORCHSIG: "torchsig_dataset.get_torchsig_loaders",
    DatasetType.PANORADIO: "panoradio_dataset.get_panoradio_loaders",
}
