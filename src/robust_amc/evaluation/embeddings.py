"""Embedding analysis and visualization utilities."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def compute_cluster_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute clustering quality metrics for embeddings.

    Args:
        embeddings: Embedding array of shape (n_samples, embedding_dim)
        labels: Label array of shape (n_samples,)

    Returns:
        Dict with silhouette score, inter/intra class distances
    """
    # Subsample for silhouette score computation (it's expensive)
    n_samples = min(5000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    emb_sample = embeddings[indices]
    lab_sample = labels[indices]

    # Silhouette score
    sil_score = silhouette_score(emb_sample, lab_sample)

    # Inter-class and intra-class distances
    unique_labels = np.unique(labels)
    class_centroids = []
    intra_distances = []

    for label in unique_labels:
        mask = labels == label
        class_emb = embeddings[mask]
        centroid = class_emb.mean(axis=0)
        class_centroids.append(centroid)

        # Intra-class: mean distance to centroid
        intra_dist = np.mean(np.linalg.norm(class_emb - centroid, axis=1))
        intra_distances.append(intra_dist)

    class_centroids = np.array(class_centroids)
    inter_distances = cdist(class_centroids, class_centroids)

    # Average inter-class distance (excluding diagonal)
    mask = ~np.eye(len(unique_labels), dtype=bool)
    avg_inter = inter_distances[mask].mean()
    avg_intra = np.mean(intra_distances)

    return {
        "silhouette_score": float(sil_score),
        "avg_inter_class_distance": float(avg_inter),
        "avg_intra_class_distance": float(avg_intra),
        "inter_intra_ratio": float(avg_inter / (avg_intra + 1e-8)),
    }


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    title: str = "t-SNE Embedding Visualization",
    figsize: tuple[int, int] = (12, 10),
    perplexity: int = 30,
    n_samples: int = 5000,
    ax: Optional[plt.Axes] = None,
    verbose: bool = True,
) -> plt.Figure:
    """Plot 2D t-SNE visualization of embeddings.

    Args:
        embeddings: Embedding array of shape (n_samples, embedding_dim)
        labels: Label array of shape (n_samples,)
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
        n_samples: Max samples to plot (for speed)
        ax: Optional existing axes to plot on
        verbose: Whether to print progress

    Returns:
        Matplotlib figure
    """
    # Subsample if too many points
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Run t-SNE
    if verbose:
        print(f"  Running t-SNE on {len(embeddings)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20.colors

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i % len(colors)]],
            label=class_names[label] if label < len(class_names) else str(label),
            alpha=0.6,
            s=10,
        )

    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    plt.tight_layout()
    return fig


def plot_embeddings_by_snr(
    embeddings: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    class_names: list[str],
    title_prefix: str = "",
    figsize: tuple[int, int] = (18, 5),
    verbose: bool = True,
) -> plt.Figure:
    """Plot t-SNE embeddings at different SNR ranges.

    Helps diagnose if representations degrade at low SNR.

    Args:
        embeddings: Embedding array of shape (n_samples, embedding_dim)
        labels: Label array of shape (n_samples,)
        snrs: SNR values array of shape (n_samples,)
        class_names: List of class names
        title_prefix: Prefix for the plot title
        figsize: Figure size
        verbose: Whether to print progress

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    snr_ranges = [
        (float("-inf"), -10, "SNR < -10 dB"),
        (-10, 0, "-10 <= SNR < 0 dB"),
        (0, 10, "0 <= SNR < 10 dB"),
        (10, float("inf"), "SNR >= 10 dB"),
    ]

    colors = plt.cm.tab20.colors

    for ax, (low, high, range_name) in zip(axes, snr_ranges):
        mask = (snrs >= low) & (snrs < high)
        if mask.sum() < 50:
            ax.text(0.5, 0.5, "Not enough samples", ha="center", va="center")
            ax.set_title(range_name)
            continue

        emb_subset = embeddings[mask]
        lab_subset = labels[mask]

        # Subsample for speed
        if len(emb_subset) > 2000:
            indices = np.random.choice(len(emb_subset), 2000, replace=False)
            emb_subset = emb_subset[indices]
            lab_subset = lab_subset[indices]

        if verbose:
            print(f"  Running t-SNE for {range_name} ({len(emb_subset)} samples)...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(emb_subset) // 4),
            random_state=42,
            n_jobs=-1,
        )
        emb_2d = tsne.fit_transform(emb_subset)

        for i, label in enumerate(np.unique(lab_subset)):
            lmask = lab_subset == label
            ax.scatter(
                emb_2d[lmask, 0],
                emb_2d[lmask, 1],
                c=[colors[label % len(colors)]],
                alpha=0.5,
                s=5,
            )

        ax.set_title(range_name)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    plt.suptitle(f"{title_prefix} Embeddings by SNR", fontsize=14)
    plt.tight_layout()
    return fig