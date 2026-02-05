"""Cross-dataset evaluation utilities for domain shift analysis.

This module provides utilities for evaluating models across different datasets
(TorchSig synthetic → Panoradio real-world) using the family label abstraction.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from robust_amc.data import Compose, PowerNormalize
from robust_amc.data.transforms import ToTensor
from robust_amc.evaluation import accuracy_by_snr


def evaluate_family_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    family_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Evaluate a model trained on family labels.

    Args:
        model: The trained model
        data_loader: DataLoader yielding (x, y, snr) batches
        device: Device to use for inference
        family_names: Optional list of family names for the confusion matrix

    Returns:
        Dictionary with:
            - accuracy: Overall accuracy
            - snr_accuracy: Dict mapping SNR to accuracy
            - predictions: Array of predicted labels
            - targets: Array of true labels
            - snrs: Array of SNR values
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_snrs = []

    with torch.no_grad():
        for batch in data_loader:
            x, y, snr = batch
            x = x.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            all_snrs.extend(snr.numpy())

    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    snrs_arr = np.array(all_snrs)

    # Compute metrics
    accuracy = (predictions == targets).mean()
    snr_acc = accuracy_by_snr(targets, predictions, snrs_arr)

    result = {
        "accuracy": float(accuracy),
        "snr_accuracy": snr_acc,
        "predictions": predictions,
        "targets": targets,
        "snrs": snrs_arr,
    }

    if family_names:
        result["family_names"] = family_names

    return result


def evaluate_cross_domain(
    model: torch.nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    device: str,
    family_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Evaluate cross-domain performance (source vs target).

    Args:
        model: The trained model
        source_loader: DataLoader for source domain (e.g., TorchSig)
        target_loader: DataLoader for target domain (e.g., Panoradio)
        device: Device to use for inference
        family_names: Optional list of family names

    Returns:
        Dictionary with:
            - source: Evaluation results on source domain
            - target: Evaluation results on target domain
            - domain_gap: Accuracy drop from source to target
    """
    source_results = evaluate_family_model(model, source_loader, device, family_names)
    target_results = evaluate_family_model(model, target_loader, device, family_names)

    domain_gap = source_results["accuracy"] - target_results["accuracy"]

    return {
        "source": source_results,
        "target": target_results,
        "domain_gap": float(domain_gap),
    }


def evaluate_ood_gap(
    model: torch.nn.Module,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    device: str,
    family_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Evaluate in-distribution vs out-of-distribution performance.

    Args:
        model: The trained model
        id_loader: DataLoader for in-distribution test data
        ood_loader: DataLoader for out-of-distribution test data
        device: Device to use for inference
        family_names: Optional list of family names

    Returns:
        Dictionary with:
            - in_distribution: Evaluation results on ID data
            - out_of_distribution: Evaluation results on OOD data
            - ood_gap: Accuracy drop from ID to OOD
    """
    id_results = evaluate_family_model(model, id_loader, device, family_names)
    ood_results = evaluate_family_model(model, ood_loader, device, family_names)

    ood_gap = id_results["accuracy"] - ood_results["accuracy"]

    return {
        "in_distribution": id_results,
        "out_of_distribution": ood_results,
        "ood_gap": float(ood_gap),
    }


def compute_family_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_families: int,
) -> np.ndarray:
    """Compute confusion matrix for family classification.

    Args:
        predictions: Predicted family indices
        targets: True family indices
        num_families: Number of families

    Returns:
        Confusion matrix of shape (num_families, num_families)
    """
    cm = np.zeros((num_families, num_families), dtype=np.int64)
    for pred, target in zip(predictions, targets):
        cm[target, pred] += 1
    return cm


def accuracy_by_family(
    predictions: np.ndarray,
    targets: np.ndarray,
    family_names: list[str],
) -> dict[str, float]:
    """Compute per-family accuracy.

    Args:
        predictions: Predicted family indices
        targets: True family indices
        family_names: List of family names

    Returns:
        Dictionary mapping family name to accuracy
    """
    result = {}
    for i, name in enumerate(family_names):
        mask = targets == i
        if mask.sum() > 0:
            result[name] = float((predictions[mask] == targets[mask]).mean())
        else:
            result[name] = 0.0
    return result


def full_evaluation_report(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ood_loader: Optional[DataLoader],
    target_loader: Optional[DataLoader],
    device: str,
    family_names: list[str],
) -> dict[str, Any]:
    """Generate a comprehensive evaluation report.

    Args:
        model: The trained model
        train_loader: Training data loader (for sanity check)
        test_loader: In-distribution test loader
        ood_loader: Out-of-distribution test loader (optional)
        target_loader: Target domain loader (optional, e.g., Panoradio)
        device: Device to use
        family_names: List of family names

    Returns:
        Comprehensive evaluation report with all metrics
    """
    report = {
        "family_names": family_names,
        "num_families": len(family_names),
    }

    # In-distribution evaluation
    test_results = evaluate_family_model(model, test_loader, device, family_names)
    report["test"] = {
        "accuracy": test_results["accuracy"],
        "snr_accuracy": test_results["snr_accuracy"],
        "per_family_accuracy": accuracy_by_family(
            test_results["predictions"],
            test_results["targets"],
            family_names,
        ),
        "confusion_matrix": compute_family_confusion_matrix(
            test_results["predictions"],
            test_results["targets"],
            len(family_names),
        ).tolist(),
    }

    # OOD evaluation
    if ood_loader is not None:
        ood_results = evaluate_ood_gap(model, test_loader, ood_loader, device, family_names)
        report["ood"] = {
            "accuracy": ood_results["out_of_distribution"]["accuracy"],
            "ood_gap": ood_results["ood_gap"],
            "snr_accuracy": ood_results["out_of_distribution"]["snr_accuracy"],
        }

    # Cross-domain evaluation
    if target_loader is not None:
        domain_results = evaluate_cross_domain(
            model, test_loader, target_loader, device, family_names
        )
        report["cross_domain"] = {
            "target_accuracy": domain_results["target"]["accuracy"],
            "domain_gap": domain_results["domain_gap"],
            "snr_accuracy": domain_results["target"]["snr_accuracy"],
        }

    return report
