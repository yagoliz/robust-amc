"""Label mapping utilities for modulation family abstraction.

This module provides functionality to map dataset-specific modulation labels
to shared modulation families (PSK, FSK, AM, SSB, QAM, OTHER) for cross-dataset
evaluation between TorchSig and Panoradio.
"""

import fnmatch
import re
from pathlib import Path
from typing import Optional

import yaml


class FamilyMapper:
    """Maps dataset-specific labels to modulation families.

    This class loads a YAML configuration file that defines the mapping from
    dataset-specific modulation labels to shared family labels. It supports
    wildcard patterns for flexible matching.

    Example YAML structure:
        PSK:
          - bpsk
          - qpsk
          - 8psk
        FSK:
          - 2fsk
          - 4fsk
          - gfsk*  # wildcard for gfsk variants

    Attributes:
        family_to_labels: Mapping from family name to list of dataset labels.
        label_to_family: Mapping from dataset label to family name.
        family_names: Ordered list of family names.
        family_to_idx: Mapping from family name to integer index.
    """

    def __init__(self, yaml_path: str | Path):
        """Initialize the family mapper from a YAML configuration file.

        Args:
            yaml_path: Path to the YAML configuration file.
        """
        self.yaml_path = Path(yaml_path)
        self._load_mapping()

    def _load_mapping(self) -> None:
        """Load and parse the YAML mapping file."""
        with open(self.yaml_path) as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        self.family_to_labels: dict[str, list[str]] = {}
        self._patterns: dict[str, list[str]] = {}  # Wildcard patterns per family

        for family, labels in config.items():
            if labels is None:
                labels = []
            self.family_to_labels[family] = []
            self._patterns[family] = []

            for label in labels:
                label_str = str(label)
                if "*" in label_str or "?" in label_str:
                    # Store wildcard patterns separately
                    self._patterns[family].append(label_str)
                else:
                    self.family_to_labels[family].append(label_str.lower())

        # Create ordered family list (order from YAML)
        self.family_names = list(self.family_to_labels.keys())
        self.family_to_idx = {name: idx for idx, name in enumerate(self.family_names)}

        # Create reverse mapping (label -> family) for exact matches
        self.label_to_family: dict[str, str] = {}
        for family, labels in self.family_to_labels.items():
            for label in labels:
                self.label_to_family[label.lower()] = family

    def _match_pattern(self, label: str) -> Optional[str]:
        """Try to match a label against wildcard patterns.

        Args:
            label: The dataset label to match.

        Returns:
            The family name if a pattern matches, None otherwise.
        """
        label_lower = label.lower()
        for family, patterns in self._patterns.items():
            for pattern in patterns:
                if fnmatch.fnmatch(label_lower, pattern.lower()):
                    return family
        return None

    def get_family_name(self, dataset_label: str) -> Optional[str]:
        """Get the family name for a dataset-specific label.

        Args:
            dataset_label: The original label from the dataset.

        Returns:
            The family name, or None if the label is not mapped.
        """
        label_lower = dataset_label.lower()

        # Try exact match first
        if label_lower in self.label_to_family:
            return self.label_to_family[label_lower]

        # Try wildcard pattern matching
        return self._match_pattern(dataset_label)

    def get_family_idx(self, dataset_label: str) -> Optional[int]:
        """Get the family index for a dataset-specific label.

        Args:
            dataset_label: The original label from the dataset.

        Returns:
            The family index (0 to num_families-1), or None if not mapped.
        """
        family = self.get_family_name(dataset_label)
        if family is None:
            return None
        return self.family_to_idx[family]

    def is_mapped(self, dataset_label: str) -> bool:
        """Check if a dataset label is mapped to a family.

        Args:
            dataset_label: The original label from the dataset.

        Returns:
            True if the label maps to a family, False otherwise.
        """
        return self.get_family_name(dataset_label) is not None

    @property
    def num_families(self) -> int:
        """Return the number of modulation families."""
        return len(self.family_names)

    def get_all_labels(self) -> list[str]:
        """Get all explicitly listed labels across all families.

        Returns:
            List of all dataset labels (excluding wildcard patterns).
        """
        labels = []
        for family_labels in self.family_to_labels.values():
            labels.extend(family_labels)
        return labels

    def get_family_labels(self, family: str) -> list[str]:
        """Get all labels mapped to a specific family.

        Args:
            family: The family name.

        Returns:
            List of labels mapped to this family.
        """
        return self.family_to_labels.get(family, [])

    def __repr__(self) -> str:
        return f"FamilyMapper(families={self.family_names}, num_labels={len(self.get_all_labels())})"


def load_family_mapping(yaml_path: str | Path) -> FamilyMapper:
    """Load a family mapping from a YAML configuration file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A FamilyMapper instance.
    """
    return FamilyMapper(yaml_path)


def get_default_torchsig_mapper() -> FamilyMapper:
    """Get the default TorchSig to family mapper.

    Returns:
        FamilyMapper configured for TorchSig modulations.
    """
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "label_maps" / "torchsig_to_family.yaml"
    return FamilyMapper(config_path)


def get_default_panoradio_mapper() -> FamilyMapper:
    """Get the default Panoradio to family mapper.

    Returns:
        FamilyMapper configured for Panoradio HF modes.
    """
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "label_maps" / "panoradio_to_family.yaml"
    return FamilyMapper(config_path)