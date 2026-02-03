#!/usr/bin/env python3
"""Analyze and visualize embeddings from trained models.

Uses t-SNE to visualize learned representations and diagnose
cluster quality across different training approaches.

Usage:
    uv run python scripts/analyze_embeddings.py --model all
    uv run python scripts/analyze_embeddings.py --model baseline --wandb
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

from robust_amc.data import get_data_loaders, PowerNormalize, Compose
from robust_amc.data.transforms import ToTensor
from robust_amc.data.radioml_loader import MODULATION_CLASSES
from robust_amc.models import create_pfcnn, create_clsr_amc
from robust_amc.evaluation import get_embeddings
from robust_amc.training import WandbLogger


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

    Returns:
        Matplotlib figure
    """
    # Subsample if too many points
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Run t-SNE
    print(f"  Running t-SNE on {len(embeddings)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

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
) -> plt.Figure:
    """Plot t-SNE embeddings at different SNR ranges.

    Helps diagnose if representations degrade at low SNR.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze model embeddings")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["baseline", "mda_dmc", "clsr_amc", "all"],
        help="Which model(s) to analyze",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/RML2016.10a_dict.pkl"),
        help="Path to RadioML dataset",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Maximum number of samples for t-SNE",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="robust-amc",
        help="W&B project name",
    )
    return parser.parse_args()


def get_device(device_str: str) -> str:
    """Get the device to use."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str


def main():
    args = parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    print("=" * 60)
    print("Embedding Analysis")
    print("=" * 60)

    # Check dataset
    if not args.data_path.exists():
        print(f"Dataset not found at {args.data_path}")
        sys.exit(1)

    # Load data
    print("\n1. Loading data...")
    transform = Compose([PowerNormalize(), ToTensor()])
    loaders = get_data_loaders(
        args.data_path,
        batch_size=256,
        train_transform=transform,
        eval_transform=transform,
        num_workers=0,
    )
    print(f"   Test set: {len(loaders['test'].dataset)} samples")

    # Initialize W&B if enabled
    wandb_logger = None
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            run_name="embedding-analysis",
            config={"analysis": "embeddings", "n_samples": args.n_samples},
        )

    # Find available models
    models_to_analyze = []

    if args.model in ["baseline", "all"]:
        baseline_path = Path("checkpoints/baseline/best_model.pt")
        if baseline_path.exists():
            model = create_pfcnn(num_classes=len(MODULATION_CLASSES))
            ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            models_to_analyze.append(("Baseline", model))
        else:
            print(f"   Baseline checkpoint not found at {baseline_path}")

    if args.model in ["mda_dmc", "all"]:
        mda_path = Path("checkpoints/mda_dmc/best_model.pt")
        if mda_path.exists():
            model = create_pfcnn(num_classes=len(MODULATION_CLASSES))
            ckpt = torch.load(mda_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            models_to_analyze.append(("MDA-DMC", model))
        else:
            print(f"   MDA-DMC checkpoint not found at {mda_path}")

    if args.model in ["clsr_amc", "all"]:
        clsr_path = Path("checkpoints/clsr_amc/best_model.pt")
        if clsr_path.exists():
            model = create_clsr_amc(num_classes=len(MODULATION_CLASSES))
            ckpt = torch.load(clsr_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            models_to_analyze.append(("CLSR-AMC", model))
        else:
            print(f"   CLSR-AMC checkpoint not found at {clsr_path}")

    if not models_to_analyze:
        print("\nNo trained models found!")
        print("Run training scripts first:")
        print("  uv run python scripts/train_baseline.py")
        print("  uv run python scripts/train_mda_dmc.py")
        print("  uv run python scripts/train_clsr_amc.py")
        sys.exit(1)

    print(f"\n2. Analyzing {len(models_to_analyze)} model(s): {[n for n, _ in models_to_analyze]}")

    # Collect metrics for comparison
    all_metrics = {}

    for name, model in models_to_analyze:
        print(f"\n{'=' * 40}")
        print(f"Analyzing: {name}")
        print("=" * 40)

        # Extract embeddings
        print("  Extracting embeddings...")
        embeddings, labels, snrs = get_embeddings(model, loaders["test"], device=device)
        print(f"  Shape: {embeddings.shape}")

        # Compute cluster metrics
        print("  Computing cluster metrics...")
        metrics = compute_cluster_metrics(embeddings, labels)
        all_metrics[name] = metrics

        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Inter/Intra Ratio: {metrics['inter_intra_ratio']:.4f}")
        print(f"  Avg Inter-class Distance: {metrics['avg_inter_class_distance']:.4f}")
        print(f"  Avg Intra-class Distance: {metrics['avg_intra_class_distance']:.4f}")

        # t-SNE visualization (all data)
        print("  Generating t-SNE visualization...")
        fig = plot_embeddings_tsne(
            embeddings,
            labels,
            MODULATION_CLASSES,
            title=f"{name} Embeddings (t-SNE)",
            n_samples=args.n_samples,
        )
        save_path = args.results_dir / f"{name.lower().replace('-', '_')}_embeddings_tsne.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

        # SNR-stratified visualization
        print("  Generating SNR-stratified visualization...")
        fig = plot_embeddings_by_snr(
            embeddings, labels, snrs, MODULATION_CLASSES, title_prefix=name
        )
        save_path = args.results_dir / f"{name.lower().replace('-', '_')}_embeddings_by_snr.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

        # Log to W&B
        if wandb_logger is not None:
            wandb_logger.log_embeddings(
                embeddings, labels, MODULATION_CLASSES, snrs, n_samples=args.n_samples
            )

    # Print comparison summary
    print("\n" + "=" * 60)
    print("Cluster Quality Comparison")
    print("=" * 60)
    print(f"{'Model':<15} {'Silhouette':>12} {'Inter/Intra':>12}")
    print("-" * 41)
    for name, metrics in all_metrics.items():
        print(
            f"{name:<15} {metrics['silhouette_score']:>12.4f} "
            f"{metrics['inter_intra_ratio']:>12.4f}"
        )

    # Interpretation guide
    print("\nInterpretation:")
    print("  - Higher silhouette score = better cluster separation (range: -1 to 1)")
    print("  - Higher inter/intra ratio = more compact, well-separated clusters")
    print("  - Compare low vs high SNR visualizations to assess noise robustness")

    if wandb_logger is not None:
        wandb_logger.finish()

    print(f"\nResults saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
