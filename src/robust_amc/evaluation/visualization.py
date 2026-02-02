"""Visualization utilities for modulation classification."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_constellation(
    iq_data: np.ndarray | torch.Tensor,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.5,
    s: float = 10,
    color: str = "blue",
    figsize: tuple[int, int] = (6, 6),
) -> plt.Axes:
    """Plot I/Q constellation diagram.

    Args:
        iq_data: I/Q samples with shape (2, N) or (N, 2) or (B, 2, N)
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        alpha: Point transparency
        s: Point size
        color: Point color
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib axes object
    """
    if isinstance(iq_data, torch.Tensor):
        iq_data = iq_data.detach().cpu().numpy()

    # Handle different input shapes
    if iq_data.ndim == 3:
        # Batch of samples: (B, 2, N) -> flatten to (2, B*N)
        B, C, N = iq_data.shape
        iq_data = iq_data.transpose(1, 0, 2).reshape(2, -1)
    elif iq_data.ndim == 2:
        if iq_data.shape[0] != 2:
            # Assume (N, 2) -> transpose to (2, N)
            iq_data = iq_data.T

    i_data = iq_data[0]
    q_data = iq_data[1]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.scatter(i_data, q_data, alpha=alpha, s=s, c=color)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    return ax


def plot_constellation_grid(
    samples_dict: dict[str, np.ndarray | torch.Tensor],
    n_cols: int = 4,
    figsize_per_plot: tuple[float, float] = (3, 3),
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """Plot multiple constellations in a grid.

    Args:
        samples_dict: Dictionary mapping modulation names to I/Q samples
        n_cols: Number of columns in grid
        figsize_per_plot: Size of each subplot
        suptitle: Overall figure title

    Returns:
        Matplotlib figure
    """
    n_plots = len(samples_dict)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        squeeze=False,
    )

    for idx, (name, data) in enumerate(samples_dict.items()):
        row, col = idx // n_cols, idx % n_cols
        plot_constellation(data, title=name, ax=axes[row, col])

    # Hide empty subplots
    for idx in range(n_plots, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()
    return fig


def plot_accuracy_vs_snr(
    snr_values: np.ndarray | list,
    accuracies: dict[str, np.ndarray | list],
    title: str = "Accuracy vs SNR",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (10, 6),
    show_legend: bool = True,
) -> plt.Axes:
    """Plot accuracy vs SNR curves for multiple models.

    Args:
        snr_values: Array of SNR values in dB
        accuracies: Dictionary mapping model names to accuracy arrays
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        figsize: Figure size if creating new figure
        show_legend: Whether to show legend

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for idx, (name, acc) in enumerate(accuracies.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(
            snr_values, acc,
            label=name,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    if show_legend:
        ax.legend(loc="lower right")

    return ax


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    normalize: bool = True,
    show_values: bool = True,
    fmt: str = ".2f",
) -> plt.Axes:
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix of shape (n_classes, n_classes)
        class_names: List of class names
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        figsize: Figure size if creating new figure
        cmap: Colormap name
        normalize: Whether to normalize rows to sum to 1
        show_values: Whether to show values in cells
        fmt: Format string for values

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(
        cm,
        annot=show_values,
        fmt=fmt if normalize else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    return ax


def plot_impairment_sweep(
    impairment_values: np.ndarray | list,
    accuracies: dict[str, np.ndarray | list],
    impairment_name: str,
    impairment_unit: str = "",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Axes:
    """Plot accuracy vs impairment magnitude.

    Args:
        impairment_values: Array of impairment magnitudes
        accuracies: Dictionary mapping model names to accuracy arrays
        impairment_name: Name of the impairment (e.g., "CFO")
        impairment_unit: Unit of impairment (e.g., "normalized")
        title: Plot title (auto-generated if None)
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"Accuracy vs {impairment_name}"

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v"]

    for idx, (name, acc) in enumerate(accuracies.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(
            impairment_values, acc,
            label=name,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
        )

    xlabel = f"{impairment_name}"
    if impairment_unit:
        xlabel += f" ({impairment_unit})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower left")

    return ax


def plot_training_history(
    history: dict[str, list],
    title: str = "Training History",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss',
                 'train_acc', 'val_acc'
        title: Overall figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig
