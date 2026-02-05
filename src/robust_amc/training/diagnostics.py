"""Training diagnostics and comparison utilities."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch


def load_history_from_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load training history from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Training history dictionary or None if not found
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("history", {})


def plot_multi_run_comparison(
    histories: dict[str, dict],
    metric: str = "val_acc",
    title: Optional[str] = None,
    figsize: tuple[int, int] = (10, 6),
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot multiple training runs on same axes for comparison.

    Args:
        histories: Dictionary mapping run names to history dicts
        metric: Metric key to plot (e.g., 'val_acc', 'train_loss')
        title: Plot title (default: auto-generated from metric)
        figsize: Figure size
        ax: Optional existing axes to plot on

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for name, history in histories.items():
        if metric in history and len(history[metric]) > 0:
            ax.plot(history[metric], label=name, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Comparison: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_loss_component_breakdown(
    history: dict,
    figsize: tuple[int, int] = (12, 5),
) -> Figure:
    """Plot stacked area chart of loss components for CLSR-AMC.

    Args:
        history: Training history dict with loss components
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(len(history.get("train_contrastive", [])))

    if len(epochs) == 0:
        return fig

    # Left: Absolute values
    ax = axes[0]
    if "train_contrastive" in history:
        ax.plot(epochs, history["train_contrastive"], label="Contrastive", linewidth=2)
    if "train_reconstruction" in history:
        ax.plot(epochs, history["train_reconstruction"], label="Reconstruction", linewidth=2)
    if "train_classification" in history:
        ax.plot(epochs, history["train_classification"], label="Classification", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.set_title("Loss Components (Absolute)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Ratios (proportion of total)
    ax = axes[1]
    con = np.array(history.get("train_contrastive", [0]))
    rec = np.array(history.get("train_reconstruction", [0]))
    cls = np.array(history.get("train_classification", [0]))
    total = con + rec + cls + 1e-8

    ax.stackplot(
        epochs,
        con / total,
        rec / total,
        cls / total,
        labels=["Contrastive", "Reconstruction", "Classification"],
        alpha=0.7,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proportion of Total Loss")
    ax.set_title("Loss Components (Proportion)")
    ax.legend(loc="upper right")
    ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig


def plot_comprehensive_diagnostic(
    histories: dict[str, dict],
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (16, 12),
) -> Figure:
    """Create comprehensive 6-panel diagnostic figure.

    Panels include:
    - (a) Validation Accuracy
    - (b) Training Loss
    - (c) Validation Loss
    - (d) Learning Rate Schedule
    - (e) Gradient Norms
    - (f) Generalization Gap

    Args:
        histories: Dictionary mapping run names to history dicts
        save_path: Optional path to save the figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    colors = plt.cm.tab10.colors

    # Panel 1: Validation Accuracy Comparison
    ax = axes[0, 0]
    for i, (name, history) in enumerate(histories.items()):
        if "val_acc" in history and len(history["val_acc"]) > 0:
            ax.plot(history["val_acc"], label=name, linewidth=2, color=colors[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("(a) Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Training Loss Comparison
    ax = axes[0, 1]
    for i, (name, history) in enumerate(histories.items()):
        if "train_loss" in history and len(history["train_loss"]) > 0:
            ax.plot(history["train_loss"], label=name, linewidth=2, color=colors[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("(b) Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Validation Loss Comparison
    ax = axes[0, 2]
    for i, (name, history) in enumerate(histories.items()):
        if "val_loss" in history and len(history["val_loss"]) > 0:
            ax.plot(history["val_loss"], label=name, linewidth=2, color=colors[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(c) Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Learning Rate
    ax = axes[1, 0]
    for i, (name, history) in enumerate(histories.items()):
        if "learning_rates" in history and len(history["learning_rates"]) > 0:
            ax.semilogy(history["learning_rates"], label=name, linewidth=2, color=colors[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("(d) Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: Gradient Norms
    ax = axes[1, 1]
    for i, (name, history) in enumerate(histories.items()):
        if "gradient_norms" in history and len(history["gradient_norms"]) > 0:
            ax.semilogy(history["gradient_norms"], label=name, linewidth=2, alpha=0.7, color=colors[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("(e) Gradient Norms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 6: Train-Val Gap (Generalization)
    ax = axes[1, 2]
    for i, (name, history) in enumerate(histories.items()):
        if "train_acc" in history and "val_acc" in history:
            train_acc = np.array(history["train_acc"])
            val_acc = np.array(history["val_acc"])
            if len(train_acc) > 0 and len(val_acc) > 0:
                gap = train_acc - val_acc
                ax.plot(gap, label=name, linewidth=2, color=colors[i])
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train - Val Accuracy")
    ax.set_title("(f) Generalization Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def analyze_training_diagnostics(histories: dict[str, dict]) -> dict[str, list[str]]:
    """Analyze training histories and return diagnostic insights.

    Args:
        histories: Dictionary mapping run names to history dicts

    Returns:
        Dictionary with 'warnings' and 'info' lists of diagnostic messages
    """
    warnings = []
    info = []

    # Check for large generalization gaps
    for name, history in histories.items():
        if "train_acc" in history and "val_acc" in history:
            train_acc = np.array(history["train_acc"])
            val_acc = np.array(history["val_acc"])
            if len(train_acc) > 0 and len(val_acc) > 0:
                final_gap = train_acc[-1] - val_acc[-1]
                if final_gap > 0.1:
                    warnings.append(
                        f"{name}: Large generalization gap ({final_gap:.3f}) - "
                        "may indicate overfitting"
                    )

    # Check for gradient issues
    for name, history in histories.items():
        if "gradient_norms" in history and len(history["gradient_norms"]) > 0:
            grad_norms = np.array(history["gradient_norms"])
            if np.any(grad_norms > 100):
                warnings.append(
                    f"{name}: Large gradient norms (max: {grad_norms.max():.1f}) - "
                    "may indicate training instability"
                )
            if np.any(grad_norms < 0.001):
                warnings.append(
                    f"{name}: Small gradient norms (min: {grad_norms.min():.4f}) - "
                    "may indicate vanishing gradients"
                )

    # Check for CLSR-AMC loss dominance
    for name, history in histories.items():
        if all(k in history for k in ["train_contrastive", "train_reconstruction", "train_classification"]):
            con = np.array(history["train_contrastive"])
            rec = np.array(history["train_reconstruction"])
            cls = np.array(history["train_classification"])
            if len(con) > 0:
                total = con + rec + cls + 1e-8
                con_ratio = (con / total).mean()
                if con_ratio > 0.6:
                    warnings.append(
                        f"{name}: Contrastive loss dominates ({con_ratio:.1%} of total) - "
                        "consider reducing contrastive_weight"
                    )

    # Summary info
    for name, history in histories.items():
        best_acc = history.get("best_val_acc")
        best_epoch = history.get("best_epoch")
        if best_acc is not None:
            info.append(f"{name}: Best val accuracy {best_acc:.4f} at epoch {best_epoch}")

    return {"warnings": warnings, "info": info}