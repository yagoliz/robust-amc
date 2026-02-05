#!/usr/bin/env python3
"""Train baseline PF-CNN model on RadioML2016.10a."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt

from robust_amc.data import (
    OVERLAPPING_CLASSES,
    Compose,
    PowerNormalize,
    get_data_loaders,
    get_data_loaders_2018,
    get_data_loaders_2018_fast,
    is_preprocessed_available,
)
from robust_amc.data.radioml2018_loader import MODULATION_CLASSES_2018 as CLASSES_2018
from robust_amc.data.radioml_loader import MODULATION_CLASSES as CLASSES_2016
from robust_amc.data.transforms import ToTensor
from robust_amc.evaluation import (
    compute_confusion_matrix,
    evaluate_model,
    evaluate_snr_sweep,
    plot_accuracy_vs_snr,
    plot_confusion_matrix,
    plot_training_history,
)
from robust_amc.models import create_pfcnn
from robust_amc.training import Trainer, TrainingConfig, WandbLogger
from robust_amc.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline PF-CNN")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["2016", "2018"],
        default="2016",
        help="Which RadioML dataset to train on",
    )
    parser.add_argument(
        "--data-path-2016",
        type=Path,
        default=Path("data/RML2016.10a_dict.pkl"),
        help="Path to RadioML2016.10a dataset",
    )
    parser.add_argument(
        "--data-path-2018",
        type=Path,
        default=Path("data/GOLD_XYZ_OSC.0001_1024.hdf5"),
        help="Path to RadioML2018.01a dataset",
    )
    parser.add_argument(
        "--overlapping-only",
        action="store_true",
        help="Only use classes that overlap between 2016 and 2018 (for cross-dataset eval)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="(Deprecated) Path to RadioML dataset - use --data-path-2016 or --data-path-2018",
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
        default=None,
        help="Directory to save checkpoints (default: checkpoints/baseline_{dataset})",
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
        help="Number of data loading workers (0 for macOS, 4+ for Linux/cluster)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None = non-deterministic)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set default checkpoint directory based on dataset
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(f"checkpoints/baseline_{args.dataset}")

    # Create directories
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    print("=" * 60)
    print(f"Training Baseline PF-CNN on RadioML{args.dataset}")
    print("=" * 60)
    if args.seed is not None:
        print(f"   Random seed: {args.seed}")

    # Determine data path and load data
    if args.dataset == "2016":
        data_path = args.data_path or args.data_path_2016
        if not data_path.exists():
            print(f"Dataset not found at {data_path}")
            print("Please download RadioML2016.10a")
            sys.exit(1)
        modulation_classes = CLASSES_2016
    else:  # 2018
        data_path = args.data_path or args.data_path_2018
        if not data_path.exists():
            print(f"Dataset not found at {data_path}")
            print("Please download RadioML2018.01a")
            sys.exit(1)
        modulation_classes = CLASSES_2018

    # Use overlapping classes if requested (for cross-dataset compatibility)
    if args.overlapping_only:
        modulation_classes = OVERLAPPING_CLASSES
        print(f"   Using overlapping classes only: {len(modulation_classes)} classes")

    # Set up transforms
    transform = Compose([PowerNormalize(), ToTensor()])

    # Load data
    print("\n1. Loading data...")
    if args.dataset == "2016":
        loaders = get_data_loaders(
            data_path,
            batch_size=args.batch_size,
            train_transform=transform,
            eval_transform=transform,
            num_workers=args.num_workers,
        )
    else:  # 2018
        # Use fast preprocessed loader if available
        if is_preprocessed_available():
            print("   Using preprocessed format (fast)")
            loaders = get_data_loaders_2018_fast(
                batch_size=args.batch_size,
                train_transform=transform,
                eval_transform=transform,
                num_workers=args.num_workers,
                overlapping_only=args.overlapping_only,
            )
        else:
            print("   Using HDF5 format (slow - run preprocess_radioml2018.py for faster loading)")
            loaders = get_data_loaders_2018(
                data_path,
                batch_size=args.batch_size,
                train_transform=transform,
                eval_transform=transform,
                num_workers=args.num_workers,
                split_segments=True,
                overlapping_only=args.overlapping_only,
            )
        if "class_names" in loaders:
            modulation_classes = loaders["class_names"]

    print(f"   Dataset: RadioML{args.dataset}")
    print(f"   Train: {len(loaders['train'].dataset)} samples")
    print(f"   Val:   {len(loaders['val'].dataset)} samples")
    print(f"   Test:  {len(loaders['test'].dataset)} samples")
    print(f"   Classes: {len(modulation_classes)}")

    # Create model
    print("\n2. Creating model...")
    model = create_pfcnn(num_classes=len(modulation_classes), variant="default")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Training config
    config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    print(f"   Device: {config.device}")

    # Initialize W&B logger if enabled
    wandb_logger = None
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"baseline-{args.dataset}",
            config={
                "model": "PF-CNN",
                "variant": "default",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "dataset": f"RadioML{args.dataset}",
                "num_classes": len(modulation_classes),
                "overlapping_only": args.overlapping_only,
            },
        )
        wandb_logger.log_model_summary("PF-CNN Baseline", n_params)
        print("   W&B logging enabled")

    # Train
    print("\n3. Training...")
    trainer = Trainer(model, config)
    history = trainer.fit(loaders["train"], loaders["val"], verbose=True, wandb_logger=wandb_logger)

    # Save final model
    trainer.save_checkpoint(args.checkpoint_dir / "final_model.pt")
    print(f"\n   Saved checkpoint to {args.checkpoint_dir}")

    # Plot training history
    fig = plot_training_history(
        {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_acc": history.train_acc,
            "val_acc": history.val_acc,
        },
        title="PF-CNN Baseline Training",
    )
    fig.savefig(args.results_dir / "baseline_training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Evaluate on test set
    print("\n4. Evaluating on test set...")

    # Load best model
    trainer.load_checkpoint(args.checkpoint_dir / "best_model.pt")

    results = evaluate_model(model, loaders["test"], device=config.device)
    print(f"   Test Accuracy: {results['accuracy']:.4f}")

    # SNR sweep
    print("\n5. Computing accuracy vs SNR...")
    snr_values, accuracies = evaluate_snr_sweep(model, loaders["test"], device=config.device)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_accuracy_vs_snr(
        snr_values,
        {"PF-CNN Baseline": accuracies},
        title="Baseline PF-CNN: Accuracy vs SNR (AWGN)",
        ax=ax,
    )
    fig.savefig(args.results_dir / "baseline_accuracy_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("   SNR (dB) | Accuracy")
    print("   " + "-" * 20)
    for snr, acc in zip(snr_values, accuracies):
        print(f"   {snr:6d}   | {acc:.4f}")

    # Confusion matrix
    print("\n6. Computing confusion matrix...")
    cm = compute_confusion_matrix(results["targets"], results["predictions"])

    fig, ax = plt.subplots(figsize=(12, 10))
    plot_confusion_matrix(
        cm,
        modulation_classes,
        title="Baseline PF-CNN Confusion Matrix (All SNRs)",
        ax=ax,
    )
    fig.savefig(args.results_dir / "baseline_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix at high SNR
    high_snr_mask = results["snrs"] >= 10
    cm_high_snr = compute_confusion_matrix(
        results["targets"][high_snr_mask],
        results["predictions"][high_snr_mask],
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    plot_confusion_matrix(
        cm_high_snr,
        modulation_classes,
        title="Baseline PF-CNN Confusion Matrix (SNR >= 10 dB)",
        ax=ax,
    )
    fig.savefig(
        args.results_dir / "baseline_confusion_matrix_high_snr.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Log final metrics to W&B
    if wandb_logger is not None:
        wandb_logger.log_snr_accuracy(snr_values, accuracies, "baseline")
        wandb_logger.log_confusion_matrix(
            results["targets"],
            results["predictions"],
            modulation_classes,
            title="Baseline Confusion Matrix",
        )
        wandb_logger.log_image(
            "training_history",
            str(args.results_dir / "baseline_training_history.png"),
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
