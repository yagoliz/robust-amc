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
from robust_amc.training import Trainer, TrainingConfig
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
        "--checkpoint-dir",
        type=str,
        default="checkpoints/pfcnn_torchsig",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for results",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PF-CNN Training on TorchSig")
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

    # Load data
    print(f"\nLoading data from {config.data_path}")
    loaders = get_loaders(config, train_transform=train_transform, eval_transform=eval_transform)

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

    # Train
    print("\nStarting training...")
    history = trainer.fit(loaders["train"], loaders["val"])

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

    results_path = results_dir / f"pfcnn_torchsig_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    print("\nGenerating plots...")

    # Accuracy vs SNR
    fig_snr = plot_accuracy_vs_snr(test_results["snr_accuracy"])
    fig_snr.savefig(results_dir / f"accuracy_vs_snr_{timestamp}.png", dpi=150)
    print(f"  Saved accuracy vs SNR plot")

    # Confusion matrix
    from robust_amc.evaluation.cross_dataset import compute_family_confusion_matrix
    cm = compute_family_confusion_matrix(
        test_results["predictions"],
        test_results["targets"],
        num_families,
    )
    fig_cm = plot_confusion_matrix(cm, family_names, title="Family Confusion Matrix")
    fig_cm.savefig(results_dir / f"confusion_matrix_{timestamp}.png", dpi=150)
    print(f"  Saved confusion matrix")

    print("\nDone!")


if __name__ == "__main__":
    main()
