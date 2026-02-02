"""Evaluation metrics for modulation classification."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "auto",
) -> dict[str, float | np.ndarray]:
    """Evaluate model and compute metrics.

    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with 'accuracy', 'predictions', 'targets', 'snrs'
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_snrs = []

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        x, y, snr = batch
        x = x.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(y.numpy())
        all_snrs.extend(snr.numpy())

    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    snrs = np.array(all_snrs)

    accuracy = accuracy_score(targets, predictions)

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "targets": targets,
        "snrs": snrs,
    }


def compute_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        targets: True labels
        predictions: Predicted labels
        normalize: Whether to normalize rows to sum to 1

    Returns:
        Confusion matrix of shape (n_classes, n_classes)
    """
    cm = confusion_matrix(targets, predictions)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    return cm


def accuracy_by_snr(
    targets: np.ndarray,
    predictions: np.ndarray,
    snrs: np.ndarray,
) -> dict[int, float]:
    """Compute accuracy at each SNR level.

    Args:
        targets: True labels
        predictions: Predicted labels
        snrs: SNR values for each sample

    Returns:
        Dictionary mapping SNR (dB) to accuracy
    """
    unique_snrs = sorted(np.unique(snrs).astype(int))
    snr_accuracy = {}

    for snr in unique_snrs:
        mask = snrs == snr
        if mask.sum() > 0:
            acc = accuracy_score(targets[mask], predictions[mask])
            snr_accuracy[int(snr)] = acc

    return snr_accuracy


def accuracy_by_class(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict[str | int, float]:
    """Compute per-class accuracy.

    Args:
        targets: True labels
        predictions: Predicted labels
        class_names: Optional list of class names

    Returns:
        Dictionary mapping class name/index to accuracy
    """
    unique_classes = sorted(np.unique(targets))
    class_accuracy = {}

    for cls in unique_classes:
        mask = targets == cls
        if mask.sum() > 0:
            acc = accuracy_score(targets[mask], predictions[mask])
            key = class_names[cls] if class_names else cls
            class_accuracy[key] = acc

    return class_accuracy


@torch.no_grad()
def evaluate_snr_sweep(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "auto",
) -> tuple[list[int], list[float]]:
    """Evaluate model across SNR levels.

    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run on

    Returns:
        Tuple of (snr_values, accuracies)
    """
    results = evaluate_model(model, data_loader, device)
    snr_acc = accuracy_by_snr(
        results["targets"],
        results["predictions"],
        results["snrs"],
    )

    snr_values = sorted(snr_acc.keys())
    accuracies = [snr_acc[snr] for snr in snr_values]

    return snr_values, accuracies


@torch.no_grad()
def get_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract embeddings from model for visualization.

    Args:
        model: Model with get_embeddings method
        data_loader: Data loader
        device: Device to run on

    Returns:
        Tuple of (embeddings, labels, snrs)
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []
    all_snrs = []

    for batch in tqdm(data_loader, desc="Extracting embeddings", leave=False):
        x, y, snr = batch
        x = x.to(device)

        # Use get_embeddings if available, otherwise use forward hooks
        if hasattr(model, "get_embeddings"):
            emb = model.get_embeddings(x)
        else:
            emb = model(x)  # Fallback to logits

        all_embeddings.append(emb.cpu().numpy())
        all_labels.extend(y.numpy())
        all_snrs.extend(snr.numpy())

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    snrs = np.array(all_snrs)

    return embeddings, labels, snrs
