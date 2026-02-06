#!/usr/bin/env python3
"""Train PF-CNN on TorchSig synthetic data with family labels.

This script trains a Phase-Feature CNN model on TorchSig-generated synthetic
signals using modulation family labels (PSK, FSK, AM, SSB, QAM).

Example usage:
    # Train with default settings
    uv run python scripts/train_pfcnn.py

    # Train with custom config
    uv run python scripts/train_pfcnn.py --config configs/datasets/torchsig_train.yaml

    # Train with augmentations
    uv run python scripts/train_pfcnn.py --augment

    # Train with W&B tracking
    uv run python scripts/train_pfcnn.py --wandb
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from robust_amc.data import (
    Compose,
    PowerNormalize,
    get_loaders,
    load_config_from_yaml,
)
from robust_amc.data.augmentations import MDADMCPipeline
from robust_amc.data.transforms import ToTensor
from robust_amc.evaluation import (
    evaluate_family_model,
    plot_accuracy_vs_snr,
    plot_confusion_matrix,
)
from robust_amc.models import create_pfcnn
from robust_amc.training import Trainer, TrainingConfig, WandbLogger
from robust_amc.utils.device import SELECTED_DEVICE, get_device
from robust_amc.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train PF-CNN on TorchSig")

    # Data arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/torchsig_train.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )

    # Model arguments
    parser.add_argument(
        "--variant",
        type=str,
        default="default",
        choices=["small", "default", "large"],
        help="Model variant",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # Augmentation
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use MDA-DMC augmentations during training",
    )

    # Output
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for checkpoints/results (auto-generated from args if not set)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for checkpoints (default: checkpoints/<run-name>)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for results",
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases experiment tracking",
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
        help="W&B run name (auto-generated if not set)",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")

    return parser.parse_args()


def build_run_name(args) -> str:
    """Build a descriptive run name from training arguments."""
    parts = [f"pfcnn-{args.variant}"]
    if args.augment:
        parts.append("mda")
    if args.lr != 1e-3:
        parts.append(f"lr{args.lr}")
    if args.batch_size != 256:
        parts.append(f"bs{args.batch_size}")
    return "_".join(parts)


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Build run name
    run_name = args.run_name or build_run_name(args)

    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PF-CNN Training on TorchSig")
    print(f"Run: {run_name}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("=" * 60)

    # Load dataset config
    print(f"\nLoading dataset config from {args.config}")
    config = load_config_from_yaml(args.config)

    if args.data_dir:
        config.data_path = args.data_dir

    # Create transforms
    eval_transform = Compose([PowerNormalize(), ToTensor()])

    if args.augment:
        print("Using MDA-DMC augmentations")
        train_transform = Compose([
            MDADMCPipeline(p=0.5),
            PowerNormalize(),
            ToTensor(),
        ])
    else:
        train_transform = eval_transform

    # Set the device for the whole training
    device = get_device(args.device)

    # Load data
    print(f"\nLoading data from {config.data_path}")
    loaders = get_loaders(config, train_transform=train_transform, eval_transform=eval_transform, device=device)

    family_names = loaders["family_names"]
    num_families = len(family_names)
    print(f"Families ({num_families}): {family_names}")
    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Val: {len(loaders['val'].dataset)} samples")
    print(f"Test: {len(loaders['test'].dataset)} samples")

    # Create model
    print(f"\nCreating PF-CNN ({args.variant}) with {num_families} output classes")
    model = create_pfcnn(variant=args.variant, num_classes=num_families)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    training_config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        seed=args.seed,
    )

    trainer = Trainer(model, training_config)
    print(f"\nTraining on device: {trainer.device}")

    # Set up W&B logger
    wandb_logger = None
    if args.wandb:
        run_name_wb = args.wandb_run_name or run_name
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            run_name=run_name_wb,
            config={
                "dataset_config": args.config,
                "variant": args.variant,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "augment": args.augment,
                "patience": args.patience,
                "seed": args.seed,
                "num_families": num_families,
                "family_names": family_names,
                "train_samples": len(loaders["train"].dataset),
                "val_samples": len(loaders["val"].dataset),
                "test_samples": len(loaders["test"].dataset),
            },
        )
        wandb_logger.log_model_summary("PF-CNN", num_params, architecture=args.variant)
        print("W&B tracking enabled")

    # Train
    print("\nStarting training...")
    history = trainer.fit(loaders["train"], loaders["val"], wandb_logger=wandb_logger)

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {history.best_val_acc:.4f} (epoch {history.best_epoch})")

    # Load best model for evaluation
    best_model_path = checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded best model from {best_model_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_family_model(
        model, loaders["test"], str(trainer.device), family_names
    )

    print(f"Test accuracy: {test_results['accuracy']:.4f}")

    # Log test results to W&B
    if wandb_logger is not None:
        wandb_logger.log_confusion_matrix(
            test_results["targets"],
            test_results["predictions"],
            family_names,
            title="Test Confusion Matrix",
        )
        snr_acc = test_results["snr_accuracy"]
        if snr_acc:
            snr_values = sorted(snr_acc.keys())
            accuracies = [snr_acc[s] for s in snr_values]
            wandb_logger.log_snr_accuracy(snr_values, accuracies, model_name="PF-CNN")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "config": {
            "dataset_config": args.config,
            "variant": args.variant,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "augment": args.augment,
            "seed": args.seed,
        },
        "family_names": family_names,
        "training": {
            "best_val_acc": history.best_val_acc,
            "best_epoch": history.best_epoch,
            "train_loss": history.train_loss,
            "train_acc": history.train_acc,
            "val_loss": history.val_loss,
            "val_acc": history.val_acc,
        },
        "test": {
            "accuracy": test_results["accuracy"],
            "snr_accuracy": test_results["snr_accuracy"],
        },
    }

    results_path = results_dir / f"{run_name}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    print("\nGenerating plots...")

    # Accuracy vs SNR
    snr_acc = test_results["snr_accuracy"]
    if snr_acc:
        snr_vals = sorted(snr_acc.keys())
        accs = [snr_acc[s] for s in snr_vals]
        ax_snr = plot_accuracy_vs_snr(snr_vals, {"PF-CNN": accs})
        ax_snr.figure.savefig(results_dir / f"{run_name}_accuracy_vs_snr_{timestamp}.png", dpi=150)
    print(f"  Saved accuracy vs SNR plot")

    # Confusion matrix
    from robust_amc.evaluation.cross_dataset import compute_family_confusion_matrix
    cm = compute_family_confusion_matrix(
        test_results["predictions"],
        test_results["targets"],
        num_families,
    )
    ax_cm = plot_confusion_matrix(cm, family_names, title="Family Confusion Matrix")
    ax_cm.figure.savefig(results_dir / f"{run_name}_confusion_matrix_{timestamp}.png", dpi=150)
    print("  Saved confusion matrix")

    # Log plots to W&B and finish
    if wandb_logger is not None:
        snr_plot = str(results_dir / f"{run_name}_accuracy_vs_snr_{timestamp}.png")
        cm_plot = str(results_dir / f"{run_name}_confusion_matrix_{timestamp}.png")
        wandb_logger.log_image("eval/accuracy_vs_snr", snr_plot)
        wandb_logger.log_image("eval/confusion_matrix", cm_plot)
        wandb_logger.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
