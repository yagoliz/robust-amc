#!/usr/bin/env python3
"""Train CLSR-AMC on TorchSig synthetic data with family labels.

CLSR-AMC uses joint multi-task learning with three loss components:
1. Contrastive (NT-Xent) between augmented views
2. Self-reconstruction of the original signal
3. Classification into modulation families

Example usage:
    # Train with default settings (joint multi-task)
    uv run python scripts/train_clsr_amc.py

    # Contrastive pretraining only (no classification)
    uv run python scripts/train_clsr_amc.py --classification-weight 0

    # Train with W&B tracking
    uv run python scripts/train_clsr_amc.py --wandb
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from robust_amc.data.augmentations import MDADMCPipeline
from robust_amc.data.contrastive import get_contrastive_loaders
from robust_amc.data.registry import load_config_from_yaml
from robust_amc.data.transforms import Compose, PowerNormalize, ToTensor
from robust_amc.evaluation import (
    evaluate_family_model,
    plot_accuracy_vs_snr,
    plot_confusion_matrix,
)
from robust_amc.evaluation.cross_dataset import compute_family_confusion_matrix
from robust_amc.models import CLSRAMCLoss, create_clsr_amc
from robust_amc.training import CLSRAMCTrainer, WandbLogger
from robust_amc.utils.device import get_device
from robust_amc.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLSR-AMC on TorchSig")

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

    # Loss weights
    parser.add_argument(
        "--contrastive-weight", type=float, default=1.0,
        help="Weight for contrastive (NT-Xent) loss",
    )
    parser.add_argument(
        "--reconstruction-weight", type=float, default=1.0,
        help="Weight for reconstruction loss",
    )
    parser.add_argument(
        "--classification-weight", type=float, default=1.0,
        help="Weight for classification loss (set to 0 for pretraining)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="Temperature for NT-Xent contrastive loss",
    )

    # Augmentation
    parser.add_argument(
        "--augment-p", type=float, default=0.5,
        help="Per-augmentation probability in MDA-DMC pipeline",
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
    parts = [f"clsr_amc-{args.variant}"]
    if args.classification_weight == 0:
        parts.append("pretrain")
    if args.lr != 1e-3:
        parts.append(f"lr{args.lr}")
    if args.batch_size != 256:
        parts.append(f"bs{args.batch_size}")
    if args.temperature != 0.5:
        parts.append(f"t{args.temperature}")
    return "_".join(parts)


def main():
    args = parse_args()

    set_seed(args.seed)

    # Build run name
    run_name = args.run_name or build_run_name(args)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CLSR-AMC Training on TorchSig")
    print(f"Run: {run_name}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("=" * 60)

    # Load dataset config
    print(f"\nLoading dataset config from {args.config}")
    config = load_config_from_yaml(args.config)
    if args.data_dir:
        config.data_path = args.data_dir

    device = get_device(args.device)

    # Build augmentation and clean transform
    augmentation = MDADMCPipeline(p=args.augment_p)
    clean_transform = Compose([PowerNormalize(), ToTensor()])

    # Load contrastive data (train/val = 5-tuple, test = 3-tuple)
    extra = config.extra_config
    print(f"\nLoading data from {config.data_path}")
    loaders = get_contrastive_loaders(
        cache_dir=config.data_path,
        batch_size=args.batch_size,
        augmentation=augmentation,
        clean_transform=clean_transform,
        crop_length=extra.get("crop_length", 128),
        train_ratio=extra.get("train_ratio", 0.6),
        val_ratio=extra.get("val_ratio", 0.2),
        test_ratio=extra.get("test_ratio", 0.2),
        num_workers=args.num_workers,
        seed=config.seed,
        generate_if_missing=extra.get("generate_if_missing", True),
        generation_config=extra.get("generation_config"),
        device=device,
    )

    family_names = loaders["family_names"]
    num_families = len(family_names)
    print(f"Families ({num_families}): {family_names}")
    print(f"Train: {len(loaders['train'].dataset)} samples")
    print(f"Val: {len(loaders['val'].dataset)} samples")
    print(f"Test: {len(loaders['test'].dataset)} samples")

    # Create model
    seq_len = extra.get("crop_length", 128)
    print(f"\nCreating CLSR-AMC ({args.variant}) with {num_families} output classes, seq_len={seq_len}")
    model = create_clsr_amc(num_classes=num_families, variant=args.variant, seq_len=seq_len)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create loss
    criterion = CLSRAMCLoss(
        contrastive_weight=args.contrastive_weight,
        reconstruction_weight=args.reconstruction_weight,
        classification_weight=args.classification_weight,
        temperature=args.temperature,
    )
    print(f"Loss weights: contrastive={args.contrastive_weight}, "
          f"reconstruction={args.reconstruction_weight}, "
          f"classification={args.classification_weight}")

    # Create trainer
    trainer = CLSRAMCTrainer(
        model=model,
        criterion=criterion,
        device=torch.device(device),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=args.patience,
    )
    print(f"Training on device: {device}")

    # Set up W&B logger
    wandb_logger = None
    if args.wandb:
        run_name_wb = args.wandb_run_name or run_name
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            run_name=run_name_wb,
            config={
                "model": "CLSR-AMC",
                "dataset_config": args.config,
                "variant": args.variant,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "seed": args.seed,
                "augment_p": args.augment_p,
                "contrastive_weight": args.contrastive_weight,
                "reconstruction_weight": args.reconstruction_weight,
                "classification_weight": args.classification_weight,
                "temperature": args.temperature,
                "num_families": num_families,
                "family_names": family_names,
                "train_samples": len(loaders["train"].dataset),
                "val_samples": len(loaders["val"].dataset),
                "test_samples": len(loaders["test"].dataset),
            },
        )
        wandb_logger.log_model_summary("CLSR-AMC", num_params, architecture=args.variant)
        print("W&B tracking enabled")

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        loaders["train"],
        loaders["val"],
        epochs=args.epochs,
        wandb_logger=wandb_logger,
    )

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {history.best_val_acc:.4f} (epoch {history.best_epoch})")

    # Load best model for evaluation
    best_model_path = checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        trainer.load_checkpoint(best_model_path)
        print(f"Loaded best model from {best_model_path}")

    # Evaluate on test set (standard 3-tuple loader, model.forward returns logits)
    print("\nEvaluating on test set...")
    test_results = evaluate_family_model(
        model, loaders["test"], device, family_names
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
            wandb_logger.log_snr_accuracy(snr_values, accuracies, model_name="CLSR-AMC")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "config": {
            "model": "CLSR-AMC",
            "dataset_config": args.config,
            "variant": args.variant,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "augment_p": args.augment_p,
            "contrastive_weight": args.contrastive_weight,
            "reconstruction_weight": args.reconstruction_weight,
            "classification_weight": args.classification_weight,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "family_names": family_names,
        "training": {
            "best_val_acc": history.best_val_acc,
            "best_epoch": history.best_epoch,
            "train_loss": history.train_loss,
            "train_contrastive": history.train_contrastive,
            "train_reconstruction": history.train_reconstruction,
            "train_classification": history.train_classification,
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

    snr_acc = test_results["snr_accuracy"]
    if snr_acc:
        snr_vals = sorted(snr_acc.keys())
        accs = [snr_acc[s] for s in snr_vals]
        ax_snr = plot_accuracy_vs_snr(snr_vals, {"CLSR-AMC": accs})
        ax_snr.figure.savefig(results_dir / f"{run_name}_accuracy_vs_snr_{timestamp}.png", dpi=150)
        print("  Saved accuracy vs SNR plot")

    cm = compute_family_confusion_matrix(
        test_results["predictions"],
        test_results["targets"],
        num_families,
    )
    ax_cm = plot_confusion_matrix(cm, family_names, title="CLSR-AMC Family Confusion Matrix")
    ax_cm.figure.savefig(results_dir / f"{run_name}_confusion_matrix_{timestamp}.png", dpi=150)
    print("  Saved confusion matrix")

    # Log plots to W&B and finish
    if wandb_logger is not None:
        wandb_logger.log_image("eval/accuracy_vs_snr",
                               str(results_dir / f"{run_name}_accuracy_vs_snr_{timestamp}.png"))
        wandb_logger.log_image("eval/confusion_matrix",
                               str(results_dir / f"{run_name}_confusion_matrix_{timestamp}.png"))
        wandb_logger.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()