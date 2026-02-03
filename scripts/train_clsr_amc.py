#!/usr/bin/env python3
"""Train CLSR-AMC model on RadioML2016.10a.

CLSR-AMC (Contrastive Learning with Self-Reconstruction for AMC) combines:
1. Contrastive learning between augmented views (NT-Xent loss)
2. Self-reconstruction of the original signal (MSE loss)
3. Classification of modulation type (CrossEntropy loss)

This multi-task learning approach helps the model learn robust representations
that generalize well under domain shift.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from robust_amc.data import (
    get_data_loaders,
    PowerNormalize,
    Compose,
    MDADMCPipeline,
)
from robust_amc.data.transforms import ToTensor
from robust_amc.data.radioml_loader import MODULATION_CLASSES
from robust_amc.models import create_clsr_amc, CLSRAMC
from robust_amc.models.clsr_amc import CLSRAMCLoss
from robust_amc.training import WandbLogger
from robust_amc.evaluation import (
    evaluate_model,
    evaluate_snr_sweep,
    compute_confusion_matrix,
    plot_accuracy_vs_snr,
    plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLSR-AMC model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/RML2016.10a_dict.pkl"),
        help="Path to RadioML dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/clsr_amc"),
        help="Directory to save checkpoints",
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
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to train on",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers",
    )
    # Loss weights
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=1.0,
        help="Weight for contrastive loss",
    )
    parser.add_argument(
        "--reconstruction-weight",
        type=float,
        default=0.5,
        help="Weight for reconstruction loss",
    )
    parser.add_argument(
        "--classification-weight",
        type=float,
        default=1.0,
        help="Weight for classification loss",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for NT-Xent loss",
    )
    # Augmentation
    parser.add_argument(
        "--aug-prob",
        type=float,
        default=0.5,
        help="Probability for augmentations",
    )
    # Model
    parser.add_argument(
        "--variant",
        type=str,
        default="default",
        choices=["default", "small", "large"],
        help="Model variant",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline model",
    )
    # Weights & Biases arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="robust-amc",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get the device to use for training."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class ContrastiveAugmentation:
    """Generate two augmented views for contrastive learning."""

    def __init__(self, aug_prob: float = 0.5, seed: int | None = None):
        self.aug1 = MDADMCPipeline(p=aug_prob, seed=seed)
        self.aug2 = MDADMCPipeline(p=aug_prob, seed=seed + 1 if seed else None)

    def __call__(self, x):
        """Return two augmented views."""
        return self.aug1(x), self.aug2(x)


class ContrastiveDataset(Dataset):
    """Wrapper dataset that returns two augmented views."""

    def __init__(self, base_dataset, augmentation, normalize_transform):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.normalize_transform = normalize_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, snr = self.base_dataset[idx]

        # Convert tensor back to numpy for augmentation
        if isinstance(x, torch.Tensor):
            x_np = x.numpy()
        else:
            x_np = x

        # Generate two augmented views
        x_i, x_j = self.augmentation(x_np)

        # Normalize and convert to tensor
        x_i = self.normalize_transform(x_i)
        x_j = self.normalize_transform(x_j)
        x_orig = self.normalize_transform(x_np)

        return x_i, x_j, x_orig, y, snr


@dataclass
class CLSRAMCTrainingHistory:
    """Training history for CLSR-AMC with diagnostic metrics."""
    train_loss: list = field(default_factory=list)
    train_contrastive: list = field(default_factory=list)
    train_reconstruction: list = field(default_factory=list)
    train_classification: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    # Diagnostic fields
    learning_rates: list = field(default_factory=list)
    gradient_norms: list = field(default_factory=list)


class CLSRAMCTrainer:
    """Trainer for CLSR-AMC model."""

    def __init__(
        self,
        model: CLSRAMC,
        criterion: CLSRAMCLoss,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: Path | None = None,
        early_stopping_patience: int = 15,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.history = CLSRAMCTrainingHistory()
        self.early_stopping_patience = early_stopping_patience
        self._early_stop_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_con = 0.0
        total_rec = 0.0
        total_cls = 0.0
        correct = 0
        total = 0
        gradient_norms = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            x_i, x_j, x_orig, y, _ = batch
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)
            x_orig = x_orig.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass for contrastive learning
            z_i, z_j, p_i, p_j = self.model.forward_contrastive(x_i, x_j)

            # Reconstruction from first view's embedding
            x_recon = self.model.decode(z_i)

            # Classification from first view
            logits = self.model.classify(z_i)

            # Compute losses
            losses = self.criterion(p_i, p_j, x_recon, x_orig, logits, y)

            # Backward pass
            losses["total"].backward()

            # Compute gradient norm before optimizer step
            grad_norm = self._compute_gradient_norm()
            gradient_norms.append(grad_norm)

            self.optimizer.step()

            # Track metrics
            batch_size = x_i.size(0)
            total_loss += losses["total"].item() * batch_size
            total_con += losses["contrastive"].item() * batch_size
            total_rec += losses["reconstruction"].item() * batch_size
            total_cls += losses["classification"].item() * batch_size

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += batch_size

            pbar.set_postfix(
                loss=losses["total"].item(),
                acc=correct / total,
            )

        avg_grad_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0

        return {
            "loss": total_loss / total,
            "contrastive": total_con / total,
            "reconstruction": total_rec / total,
            "classification": total_cls / total,
            "accuracy": correct / total,
            "gradient_norm": avg_grad_norm,
        }

    def _compute_gradient_norm(self) -> float:
        """Compute total gradient L2 norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict:
        """Evaluate model on validation set."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            x_i, x_j, x_orig, y, _ = batch
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)
            x_orig = x_orig.to(self.device)
            y = y.to(self.device)

            # Forward pass
            z_i, z_j, p_i, p_j = self.model.forward_contrastive(x_i, x_j)
            x_recon = self.model.decode(z_i)
            logits = self.model.classify(z_i)

            losses = self.criterion(p_i, p_j, x_recon, x_orig, logits, y)

            batch_size = x_i.size(0)
            total_loss += losses["total"].item() * batch_size

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += batch_size

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
        wandb_logger=None,
    ) -> CLSRAMCTrainingHistory:
        """Train the model."""
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history.train_loss.append(train_metrics["loss"])
            self.history.train_contrastive.append(train_metrics["contrastive"])
            self.history.train_reconstruction.append(train_metrics["reconstruction"])
            self.history.train_classification.append(train_metrics["classification"])
            self.history.train_acc.append(train_metrics["accuracy"])
            self.history.gradient_norms.append(train_metrics["gradient_norm"])

            # Validate
            val_metrics = self.evaluate(val_loader)
            self.history.val_loss.append(val_metrics["loss"])
            self.history.val_acc.append(val_metrics["accuracy"])

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            # Track learning rate
            lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(lr)

            # Logging
            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Con: {train_metrics['contrastive']:.4f} | "
                    f"Rec: {train_metrics['reconstruction']:.4f} | "
                    f"Cls: {train_metrics['classification']:.4f} | "
                    f"Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"LR: {lr:.2e}"
                )

            # Log to wandb if enabled
            if wandb_logger is not None:
                wandb_logger.log_metrics(
                    epoch=epoch + 1,
                    train_loss=train_metrics["loss"],
                    train_acc=train_metrics["accuracy"],
                    val_loss=val_metrics["loss"],
                    val_acc=val_metrics["accuracy"],
                    learning_rate=lr,
                    gradient_norm=train_metrics["gradient_norm"],
                )
                wandb_logger.log_loss_components(
                    epoch=epoch + 1,
                    contrastive=train_metrics["contrastive"],
                    reconstruction=train_metrics["reconstruction"],
                    classification=train_metrics["classification"],
                    total=train_metrics["loss"],
                )

            # Check for best model
            if val_metrics["accuracy"] > self.history.best_val_acc:
                self.history.best_val_acc = val_metrics["accuracy"]
                self.history.best_epoch = epoch + 1
                self._early_stop_counter = 0

                if self.checkpoint_dir:
                    self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
            else:
                self._early_stop_counter += 1

            # Early stopping
            if self._early_stop_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if verbose:
            print(f"\nBest validation accuracy: {self.history.best_val_acc:.4f} "
                  f"at epoch {self.history.best_epoch}")

        return self.history

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": {
                "train_loss": self.history.train_loss,
                "train_contrastive": self.history.train_contrastive,
                "train_reconstruction": self.history.train_reconstruction,
                "train_classification": self.history.train_classification,
                "train_acc": self.history.train_acc,
                "val_loss": self.history.val_loss,
                "val_acc": self.history.val_acc,
                "best_val_acc": self.history.best_val_acc,
                "best_epoch": self.history.best_epoch,
                "learning_rates": self.history.learning_rates,
                "gradient_norms": self.history.gradient_norms,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])


def main():
    args = parse_args()

    # Create directories
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training CLSR-AMC Model")
    print("=" * 60)

    # Check dataset
    if not args.data_path.exists():
        print(f"Dataset not found at {args.data_path}")
        print("Run: python scripts/download_data.py")
        sys.exit(1)

    device = get_device(args.device)
    print(f"\n1. Setup")
    print(f"   Device: {device}")
    print(f"   Loss weights: con={args.contrastive_weight}, "
          f"rec={args.reconstruction_weight}, cls={args.classification_weight}")

    # Create augmentation and transforms
    print("\n2. Loading data...")
    normalize_transform = Compose([PowerNormalize(), ToTensor()])
    contrastive_aug = ContrastiveAugmentation(aug_prob=args.aug_prob, seed=42)

    # Load base datasets
    base_loaders = get_data_loaders(
        args.data_path,
        batch_size=args.batch_size,
        train_transform=None,  # We'll apply transforms in ContrastiveDataset
        eval_transform=None,
        num_workers=args.num_workers,
    )

    # Wrap with contrastive dataset
    train_dataset = ContrastiveDataset(
        base_loaders["train"].dataset,
        contrastive_aug,
        normalize_transform,
    )
    val_dataset = ContrastiveDataset(
        base_loaders["val"].dataset,
        contrastive_aug,
        normalize_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")

    # Create model
    print(f"\n3. Creating CLSR-AMC model (variant={args.variant})...")
    model = create_clsr_amc(num_classes=len(MODULATION_CLASSES), variant=args.variant)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"   Embedding dim: {model.embedding_dim}")

    # Create loss
    criterion = CLSRAMCLoss(
        contrastive_weight=args.contrastive_weight,
        reconstruction_weight=args.reconstruction_weight,
        classification_weight=args.classification_weight,
        temperature=args.temperature,
    )

    # Initialize W&B logger if enabled
    wandb_logger = None
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            run_name=args.wandb_run_name or "clsr-amc",
            config={
                "model": "CLSR-AMC",
                "variant": args.variant,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "dataset": "RadioML2016.10a",
                "loss_weights": {
                    "contrastive": args.contrastive_weight,
                    "reconstruction": args.reconstruction_weight,
                    "classification": args.classification_weight,
                },
                "temperature": args.temperature,
                "aug_prob": args.aug_prob,
            },
        )
        wandb_logger.log_model_summary("CLSR-AMC", n_params)
        print("   W&B logging enabled")

    # Create trainer
    trainer = CLSRAMCTrainer(
        model=model,
        criterion=criterion,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    print("\n4. Training...")
    history = trainer.fit(
        train_loader, val_loader, epochs=args.epochs, verbose=True, wandb_logger=wandb_logger
    )

    # Save final model
    trainer.save_checkpoint(args.checkpoint_dir / "final_model.pt")
    print(f"\n   Saved checkpoint to {args.checkpoint_dir}")

    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total loss
    axes[0, 0].plot(history.train_loss, label="Train")
    axes[0, 0].plot(history.val_loss, label="Val")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Component losses
    axes[0, 1].plot(history.train_contrastive, label="Contrastive")
    axes[0, 1].plot(history.train_reconstruction, label="Reconstruction")
    axes[0, 1].plot(history.train_classification, label="Classification")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Loss Components")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 0].plot(history.train_acc, label="Train")
    axes[1, 0].plot(history.val_acc, label="Val")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Clear unused subplot
    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.5,
                    f"Best Val Acc: {history.best_val_acc:.4f}\n"
                    f"Best Epoch: {history.best_epoch}",
                    ha="center", va="center", fontsize=14)

    plt.suptitle("CLSR-AMC Training History", fontsize=14)
    plt.tight_layout()
    fig.savefig(args.results_dir / "clsr_amc_training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    trainer.load_checkpoint(args.checkpoint_dir / "best_model.pt")
    model.eval()

    # Standard evaluation (single forward pass)
    test_transform = normalize_transform
    test_loaders = get_data_loaders(
        args.data_path,
        batch_size=args.batch_size,
        train_transform=test_transform,
        eval_transform=test_transform,
        num_workers=args.num_workers,
    )

    results = evaluate_model(model, test_loaders["test"], device=str(device))
    print(f"   Test Accuracy: {results['accuracy']:.4f}")

    # SNR sweep
    print("\n6. Computing accuracy vs SNR...")
    snr_values, accuracies = evaluate_snr_sweep(model, test_loaders["test"], device=str(device))

    models_acc = {"CLSR-AMC": accuracies}

    print("   SNR (dB) | Accuracy")
    print("   " + "-" * 20)
    for snr, acc in zip(snr_values, accuracies):
        print(f"   {snr:6d}   | {acc:.4f}")

    # Confusion matrix
    print("\n7. Computing confusion matrix...")
    cm = compute_confusion_matrix(results["targets"], results["predictions"])

    fig, ax = plt.subplots(figsize=(12, 10))
    plot_confusion_matrix(
        cm,
        MODULATION_CLASSES,
        title="CLSR-AMC Confusion Matrix (All SNRs)",
        ax=ax,
    )
    fig.savefig(args.results_dir / "clsr_amc_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Compare with baseline if requested
    if args.compare_baseline:
        from robust_amc.models import create_pfcnn

        baseline_path = Path("checkpoints/baseline/best_model.pt")
        if baseline_path.exists():
            print("\n8. Comparing with baseline model...")
            baseline_model = create_pfcnn(num_classes=len(MODULATION_CLASSES))
            checkpoint = torch.load(baseline_path, map_location=device)
            baseline_model.load_state_dict(checkpoint["model_state_dict"])
            baseline_model.to(device)
            baseline_model.eval()

            _, baseline_acc = evaluate_snr_sweep(
                baseline_model, test_loaders["test"], device=str(device)
            )
            models_acc["PF-CNN Baseline"] = baseline_acc

            print("   Model comparison (accuracy at high SNR):")
            high_snr_idx = [i for i, s in enumerate(snr_values) if s >= 10]
            for name, accs in models_acc.items():
                high_snr_avg = sum(accs[i] for i in high_snr_idx) / len(high_snr_idx)
                print(f"   {name}: {high_snr_avg:.4f}")
        else:
            print("\n   Baseline model not found, skipping comparison.")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_accuracy_vs_snr(
        snr_values,
        models_acc,
        title="Accuracy vs SNR Comparison",
        ax=ax,
    )
    fig.savefig(args.results_dir / "clsr_amc_accuracy_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save results
    results_data = {
        "model": "CLSR-AMC",
        "variant": args.variant,
        "loss_weights": {
            "contrastive": args.contrastive_weight,
            "reconstruction": args.reconstruction_weight,
            "classification": args.classification_weight,
        },
        "temperature": args.temperature,
        "aug_prob": args.aug_prob,
        "best_val_acc": history.best_val_acc,
        "test_acc": results["accuracy"],
        "snr_accuracies": {str(s): float(a) for s, a in zip(snr_values, accuracies)},
    }

    with open(args.results_dir / "clsr_amc_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Log final metrics to W&B
    if wandb_logger is not None:
        wandb_logger.log_snr_accuracy(snr_values, accuracies, "clsr_amc")
        wandb_logger.log_confusion_matrix(
            results["targets"],
            results["predictions"],
            MODULATION_CLASSES,
            title="CLSR-AMC Confusion Matrix",
        )
        wandb_logger.log_image(
            "training_history",
            str(args.results_dir / "clsr_amc_training_history.png"),
        )
        wandb_logger.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {history.best_val_acc:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Results saved to: {args.results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
