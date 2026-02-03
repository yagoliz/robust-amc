#!/usr/bin/env python3
"""Compare training runs and generate diagnostic visualizations.

Loads checkpoints from multiple training approaches and generates
comparison plots to diagnose performance differences.

Usage:
    uv run python scripts/compare_experiments.py
    uv run python scripts/compare_experiments.py --wandb
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_amc.training import WandbLogger


def load_history_from_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load training history from checkpoint file."""
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("history", {})


def plot_multi_run_comparison(
    histories: dict[str, dict],
    metric: str = "val_acc",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot multiple training runs on same axes for comparison."""
    fig, ax = plt.subplots(figsize=figsize)

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
) -> plt.Figure:
    """Plot stacked area chart of loss components for CLSR-AMC."""
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
    save_path: Path | None = None,
    figsize: tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Create comprehensive 6-panel diagnostic figure."""
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare training runs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing checkpoints",
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


def main():
    args = parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Run Comparison")
    print("=" * 60)

    # Load histories from all checkpoints
    histories = {}

    baseline_ckpt = args.checkpoints_dir / "baseline" / "best_model.pt"
    if baseline_ckpt.exists():
        history = load_history_from_checkpoint(baseline_ckpt)
        if history:
            histories["Baseline"] = history
            print(f"  Loaded: Baseline ({len(history.get('train_loss', []))} epochs)")

    mda_ckpt = args.checkpoints_dir / "mda_dmc" / "best_model.pt"
    if mda_ckpt.exists():
        history = load_history_from_checkpoint(mda_ckpt)
        if history:
            histories["MDA-DMC"] = history
            print(f"  Loaded: MDA-DMC ({len(history.get('train_loss', []))} epochs)")

    clsr_ckpt = args.checkpoints_dir / "clsr_amc" / "best_model.pt"
    if clsr_ckpt.exists():
        history = load_history_from_checkpoint(clsr_ckpt)
        if history:
            histories["CLSR-AMC"] = history
            print(f"  Loaded: CLSR-AMC ({len(history.get('train_loss', []))} epochs)")

    if not histories:
        print("\nNo checkpoint histories found!")
        print("Run training scripts first:")
        print("  uv run python scripts/train_baseline.py")
        print("  uv run python scripts/train_mda_dmc.py")
        print("  uv run python scripts/train_clsr_amc.py")
        sys.exit(1)

    # Initialize W&B if enabled
    wandb_logger = None
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            run_name="training-comparison",
            config={"analysis": "training_comparison"},
        )

    # Generate comprehensive diagnostic figure
    print("\n1. Generating comprehensive diagnostic figure...")
    fig = plot_comprehensive_diagnostic(histories)
    save_path = args.results_dir / "training_diagnostic_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {save_path}")

    if wandb_logger:
        wandb_logger.log_image("diagnostic_comparison", str(save_path))

    # Individual metric comparisons
    print("\n2. Generating individual metric comparisons...")
    for metric in ["val_acc", "train_loss", "val_loss"]:
        fig = plot_multi_run_comparison(histories, metric=metric)
        save_path = args.results_dir / f"comparison_{metric}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {save_path}")

        if wandb_logger:
            wandb_logger.log_image(f"comparison_{metric}", str(save_path))

    # CLSR-AMC specific: loss component breakdown
    if "CLSR-AMC" in histories:
        clsr_history = histories["CLSR-AMC"]
        if "train_contrastive" in clsr_history:
            print("\n3. Generating CLSR-AMC loss breakdown...")
            fig = plot_loss_component_breakdown(clsr_history)
            save_path = args.results_dir / "clsr_amc_loss_breakdown.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"   Saved: {save_path}")

            if wandb_logger:
                wandb_logger.log_image("clsr_amc_loss_breakdown", str(save_path))

    # Summary table
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'Best Val Acc':>12} {'Best Epoch':>12} {'Final LR':>12}")
    print("-" * 53)
    for name, history in histories.items():
        best_acc = history.get("best_val_acc", "N/A")
        best_epoch = history.get("best_epoch", "N/A")
        lrs = history.get("learning_rates", [])
        final_lr = f"{lrs[-1]:.2e}" if lrs else "N/A"

        if isinstance(best_acc, float):
            best_acc = f"{best_acc:.4f}"

        print(f"{name:<15} {best_acc:>12} {best_epoch:>12} {final_lr:>12}")

    # Diagnostic insights
    print("\n" + "=" * 60)
    print("Diagnostic Insights")
    print("=" * 60)

    # Check for large generalization gaps
    for name, history in histories.items():
        if "train_acc" in history and "val_acc" in history:
            train_acc = np.array(history["train_acc"])
            val_acc = np.array(history["val_acc"])
            if len(train_acc) > 0 and len(val_acc) > 0:
                final_gap = train_acc[-1] - val_acc[-1]
                if final_gap > 0.1:
                    print(f"  [!] {name}: Large generalization gap ({final_gap:.3f})")
                    print(f"      -> May indicate overfitting or augmentation too aggressive")

    # Check for gradient issues
    for name, history in histories.items():
        if "gradient_norms" in history and len(history["gradient_norms"]) > 0:
            grad_norms = np.array(history["gradient_norms"])
            if np.any(grad_norms > 100):
                print(f"  [!] {name}: Large gradient norms detected (max: {grad_norms.max():.1f})")
                print(f"      -> May indicate training instability")
            if np.any(grad_norms < 0.001):
                print(f"  [!] {name}: Small gradient norms detected (min: {grad_norms.min():.4f})")
                print(f"      -> May indicate vanishing gradients")

    # Check for CLSR-AMC loss dominance
    if "CLSR-AMC" in histories:
        clsr = histories["CLSR-AMC"]
        if all(k in clsr for k in ["train_contrastive", "train_reconstruction", "train_classification"]):
            con = np.array(clsr["train_contrastive"])
            rec = np.array(clsr["train_reconstruction"])
            cls = np.array(clsr["train_classification"])
            if len(con) > 0:
                total = con + rec + cls + 1e-8
                con_ratio = (con / total).mean()
                if con_ratio > 0.6:
                    print(f"  [!] CLSR-AMC: Contrastive loss dominates ({con_ratio:.1%} of total)")
                    print(f"      -> Consider reducing contrastive_weight")

    if wandb_logger:
        wandb_logger.finish()

    print(f"\nResults saved to: {args.results_dir}")


if __name__ == "__main__":
    main()