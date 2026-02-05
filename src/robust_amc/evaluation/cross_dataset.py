"""Cross-dataset evaluation utilities for domain shift analysis."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from robust_amc.data import (
    CLASS_NAME_MAPPING_2018_TO_2016,
    OVERLAPPING_CLASSES,
    Compose,
    PowerNormalize,
    load_radioml2016a,
    load_radioml2018a,
)
from robust_amc.data.radioml2018_loader import MODULATION_CLASSES_2018 as CLASSES_2018
from robust_amc.data.radioml_loader import MODULATION_CLASSES as CLASSES_2016
from robust_amc.data.transforms import ToTensor
from robust_amc.evaluation import accuracy_by_snr


def get_class_mapping(
    source_classes: list[str],
    target_classes: list[str] = None,
    use_2018_mapping: bool = False,
) -> dict[int, int]:
    """Get mapping from source class indices to target (overlapping) class indices.

    Args:
        source_classes: List of class names from training dataset
        target_classes: Target class list (default: OVERLAPPING_CLASSES)
        use_2018_mapping: Whether source uses 2018 naming convention

    Returns:
        Dictionary mapping source class index to target class index
    """
    if target_classes is None:
        target_classes = OVERLAPPING_CLASSES

    mapping = {}
    for src_idx, src_name in enumerate(source_classes):
        if use_2018_mapping:
            # 2018 names need mapping via CLASS_NAME_MAPPING_2018_TO_2016
            if src_name in CLASS_NAME_MAPPING_2018_TO_2016:
                mapped_name = CLASS_NAME_MAPPING_2018_TO_2016[src_name]
                if mapped_name in target_classes:
                    mapping[src_idx] = target_classes.index(mapped_name)
        else:
            # 2016 names are used directly
            if src_name in target_classes:
                mapping[src_idx] = target_classes.index(src_name)

    return mapping


def load_overlapping_data(
    dataset: str,
    data_path_2016: Optional[Path] = None,
    data_path_2018: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load evaluation dataset with overlapping classes only.

    Returns data with class indices mapped to OVERLAPPING_CLASSES order.

    Args:
        dataset: "2016" or "2018"
        data_path_2016: Path to RadioML 2016 dataset
        data_path_2018: Path to RadioML 2018 dataset
        max_samples: Maximum samples to load (for faster testing)

    Returns:
        Tuple of (data, labels, snrs, class_names) where labels are
        indices into OVERLAPPING_CLASSES
    """
    if dataset == "2016":
        if data_path_2016 is None:
            data_path_2016 = Path("data/RML2016.10a_dict.pkl")
        if not data_path_2016.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path_2016}")

        data, labels, snrs = load_radioml2016a(data_path_2016)

        # Filter to overlapping classes and remap indices
        overlap_indices_2016 = [CLASSES_2016.index(c) for c in OVERLAPPING_CLASSES if c in CLASSES_2016]
        mask = np.isin(labels, overlap_indices_2016)
        data = data[mask]
        labels_orig = labels[mask]
        snrs = snrs[mask]

        # Remap to OVERLAPPING_CLASSES order
        old_to_new = {CLASSES_2016.index(c): OVERLAPPING_CLASSES.index(c)
                      for c in OVERLAPPING_CLASSES if c in CLASSES_2016}
        labels = np.array([old_to_new[l] for l in labels_orig])

    else:  # 2018
        if data_path_2018 is None:
            data_path_2018 = Path("data/GOLD_XYZ_OSC.0001_1024.hdf5")
        if not data_path_2018.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path_2018}")

        data, labels, snrs, class_names = load_radioml2018a(
            data_path_2018,
            split_segments=True,
            overlapping_only=True,
        )

        # Map 2018 class names to OVERLAPPING_CLASSES order
        name_2018_to_overlap = {}
        for name_2018, name_2016 in CLASS_NAME_MAPPING_2018_TO_2016.items():
            if name_2016 in OVERLAPPING_CLASSES:
                name_2018_to_overlap[name_2018] = OVERLAPPING_CLASSES.index(name_2016)

        old_to_new = {i: name_2018_to_overlap[class_names[i]]
                      for i in range(len(class_names))
                      if class_names[i] in name_2018_to_overlap}

        # Filter to classes we can map
        valid_old_indices = list(old_to_new.keys())
        mask = np.isin(labels, valid_old_indices)
        data = data[mask]
        labels_orig = labels[mask]
        snrs = snrs[mask]
        labels = np.array([old_to_new[l] for l in labels_orig])

    # Limit samples if requested
    if max_samples is not None and len(data) > max_samples:
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data[indices]
        labels = labels[indices]
        snrs = snrs[indices]

    return data, labels, snrs, OVERLAPPING_CLASSES


def evaluate_cross_dataset(
    model: torch.nn.Module,
    data: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    device: str,
    train_class_to_overlap: dict,
    batch_size: int = 256,
) -> dict:
    """Evaluate model with class mapping for cross-dataset evaluation.

    Args:
        model: The trained model
        data: Input data
        labels: Target labels (in overlapping class indices)
        snrs: SNR values
        device: Device to use
        train_class_to_overlap: Mapping from training class indices to overlapping class indices
        batch_size: Batch size

    Returns:
        Dictionary with accuracy, snr_accuracy, predictions, targets, snrs
    """
    model = model.to(device)
    model.eval()

    transform = Compose([PowerNormalize(), ToTensor()])

    all_preds = []
    all_targets = []
    all_snrs = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            batch_snrs = snrs[i:i + batch_size]

            # Transform batch
            x_batch = torch.stack([transform(x) for x in batch_data]).to(device)

            # Forward pass
            logits = model(x_batch)

            # Map model output to overlapping class indices
            # Only consider logits for overlapping classes
            overlap_indices = sorted(train_class_to_overlap.keys())
            logits_overlap = logits[:, overlap_indices]
            preds_in_overlap = logits_overlap.argmax(dim=1).cpu().numpy()
            # Map back to the overlap class index
            preds = np.array([train_class_to_overlap[overlap_indices[p]] for p in preds_in_overlap])

            all_preds.extend(preds)
            all_targets.extend(batch_labels)
            all_snrs.extend(batch_snrs)

    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    snrs_arr = np.array(all_snrs)

    # Compute metrics
    accuracy = (predictions == targets).mean()
    snr_acc = accuracy_by_snr(targets, predictions, snrs_arr)

    return {
        "accuracy": float(accuracy),
        "snr_accuracy": snr_acc,
        "predictions": predictions,
        "targets": targets,
        "snrs": snrs_arr,
    }