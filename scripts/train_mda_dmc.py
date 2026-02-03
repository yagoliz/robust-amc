#!/usr/bin/env python3
"""Train PF-CNN model with MDA-DMC augmentation on RadioML2016.10a.

This script trains the PF-CNN model with Multi-Domain Augmentation for
Domain-Mismatch Compensation (MDA-DMC) to improve robustness against
domain shifts from hardware impairments and channel variations.

MDA-DMC augmentations include:
- AGN: Additive Gaussian Noise with SNR jitter
- RSC: Rotation in Signal Constellation (phase rotation)
- SSC: Stretching in Signal Constellation (amplitude scaling)
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import torch

from robust_amc.data import (
    get_data_loaders,
    get_data_loaders_2018,
    PowerNormalize,
    Compose,
    MDADMCPipeline,
    OVERLAPPING_CLASSES,
)
from robust_amc.data.transforms import ToTensor
from robust_amc.data.radioml_loader import MODULATION_CLASSES as CLASSES_2016
from robust_amc.data.radioml2018_loader import MODULATION_CLASSES_2018 as CLASSES_2018
from robust_amc.data.impairments import CarrierFrequencyOffset, IQImbalance
from robust_amc.data.channels import RayleighFading
from robust_amc.models import create_pfcnn
from robust_amc.training import Trainer, TrainingConfig, WandbLogger
from robust_amc.evaluation import (
    evaluate_model,
    evaluate_snr_sweep,
    compute_confusion_matrix,
    plot_accuracy_vs_snr,
    plot_confusion_matrix,
    plot_training_history,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PF-CNN with MDA-DMC augmentation")
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
        help="Only use classes that overlap between 2016 and 2018",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="(Deprecated) Use --data-path-2016 or --data-path-2018",
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
        help="Directory to save checkpoints (default: checkpoints/mda_dmc_{dataset})",
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
    # MDA-DMC specific arguments
    parser.add_argument(
        "--aug-prob",
        type=float,
        default=0.5,
        help="Probability for each augmentation",
    )
    parser.add_argument(
        "--agn-snr-min",
        type=float,
        default=-5.0,
        help="Minimum SNR for AGN augmentation (dB)",
    )
    parser.add_argument(
        "--agn-snr-max",
        type=float,
        default=15.0,
        help="Maximum SNR for AGN augmentation (dB)",
    )
    parser.add_argument(
        "--rsc-angle-max",
        type=float,
        default=180.0,
        help="Maximum rotation angle for RSC augmentation (degrees)",
    )
    parser.add_argument(
        "--ssc-scale-min",
        type=float,
        default=0.8,
        help="Minimum scale for SSC augmentation",
    )
    parser.add_argument(
        "--ssc-scale-max",
        type=float,
        default=1.2,
        help="Maximum scale for SSC augmentation",
    )
    parser.add_argument(
        "--no-agn",
        action="store_true",
        help="Disable AGN augmentation",
    )
    parser.add_argument(
        "--no-rsc",
        action="store_true",
        help="Disable RSC augmentation",
    )
    parser.add_argument(
        "--no-ssc",
        action="store_true",
        help="Disable SSC augmentation",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Load baseline model and compare robustness",
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


class AugmentedTransform:
    """Transform that applies augmentation followed by normalization."""

    def __init__(self, augmentation, normalize_transform):
        self.augmentation = augmentation
        self.normalize_transform = normalize_transform

    def __call__(self, x):
        # Apply augmentation first (on raw signal)
        if self.augmentation is not None:
            x = self.augmentation(x)
        # Then normalize and convert to tensor
        return self.normalize_transform(x)


def evaluate_robustness(model, test_loader, device):
    """Evaluate model robustness under various impairments."""
    results = {}

    # 1. Baseline (clean)
    clean_results = evaluate_model(model, test_loader, device=device)
    results["clean"] = clean_results["accuracy"]

    # 2. CFO robustness
    cfo_accs = []
    for cfo_hz in [0, 500, 1000, 2000, 5000]:
        cfo_transform = Compose([
            CarrierFrequencyOffset(delta_f=cfo_hz, sample_rate=1e6),
            PowerNormalize(),
            ToTensor(),
        ])
        # Create new loader with impairment
        impaired_loader = get_data_loaders(
            test_loader.dataset.data_path,
            batch_size=test_loader.batch_size,
            train_transform=cfo_transform,
            eval_transform=cfo_transform,
            num_workers=0,
        )["test"]
        cfo_results = evaluate_model(model, impaired_loader, device=device)
        cfo_accs.append((cfo_hz, cfo_results["accuracy"]))
    results["cfo"] = cfo_accs

    # 3. I/Q imbalance robustness
    iq_accs = []
    for amp_db in [0, 1, 2, 3]:
        iq_transform = Compose([
            IQImbalance(amplitude_imbalance_db=amp_db, phase_imbalance_deg=amp_db),
            PowerNormalize(),
            ToTensor(),
        ])
        impaired_loader = get_data_loaders(
            test_loader.dataset.data_path,
            batch_size=test_loader.batch_size,
            train_transform=iq_transform,
            eval_transform=iq_transform,
            num_workers=0,
        )["test"]
        iq_results = evaluate_model(model, impaired_loader, device=device)
        iq_accs.append((amp_db, iq_results["accuracy"]))
    results["iq_imbalance"] = iq_accs

    # 4. Rayleigh fading robustness
    fading_transform = Compose([
        RayleighFading(),
        PowerNormalize(),
        ToTensor(),
    ])
    fading_loader = get_data_loaders(
        test_loader.dataset.data_path,
        batch_size=test_loader.batch_size,
        train_transform=fading_transform,
        eval_transform=fading_transform,
        num_workers=0,
    )["test"]
    fading_results = evaluate_model(model, fading_loader, device=device)
    results["rayleigh_fading"] = fading_results["accuracy"]

    return results


def main():
    args = parse_args()

    # Set default checkpoint directory based on dataset
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(f"checkpoints/mda_dmc_{args.dataset}")

    # Create directories
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Training PF-CNN with MDA-DMC on RadioML{args.dataset}")
    print("=" * 60)

    # Determine data path
    if args.dataset == "2016":
        data_path = args.data_path or args.data_path_2016
        modulation_classes = CLASSES_2016
    else:
        data_path = args.data_path or args.data_path_2018
        modulation_classes = CLASSES_2018

    if args.overlapping_only:
        modulation_classes = OVERLAPPING_CLASSES

    # Check dataset
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        sys.exit(1)

    # Create MDA-DMC augmentation pipeline
    print("\n1. Setting up MDA-DMC augmentation...")
    augmentation = MDADMCPipeline(
        agn=not args.no_agn,
        rsc=not args.no_rsc,
        ssc=not args.no_ssc,
        agn_snr_range=(args.agn_snr_min, args.agn_snr_max),
        rsc_angle_range=(-args.rsc_angle_max, args.rsc_angle_max),
        ssc_scale_range=(args.ssc_scale_min, args.ssc_scale_max),
        p=args.aug_prob,
    )
    print(f"   {augmentation}")

    # Set up transforms
    normalize_transform = Compose([PowerNormalize(), ToTensor()])
    train_transform = AugmentedTransform(augmentation, normalize_transform)
    eval_transform = normalize_transform  # No augmentation during evaluation

    # Load data
    print("\n2. Loading data...")
    if args.dataset == "2016":
        loaders = get_data_loaders(
            data_path,
            batch_size=args.batch_size,
            train_transform=train_transform,
            eval_transform=eval_transform,
            num_workers=args.num_workers,
        )
    else:
        loaders = get_data_loaders_2018(
            data_path,
            batch_size=args.batch_size,
            train_transform=train_transform,
            eval_transform=eval_transform,
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
    print("\n3. Creating model...")
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
            run_name=args.wandb_run_name or "mda-dmc",
            config={
                "model": "PF-CNN",
                "method": "MDA-DMC",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "dataset": "RadioML2016.10a",
                "augmentation": {
                    "agn": not args.no_agn,
                    "rsc": not args.no_rsc,
                    "ssc": not args.no_ssc,
                    "prob": args.aug_prob,
                    "agn_snr_range": [args.agn_snr_min, args.agn_snr_max],
                    "rsc_angle_range": [-args.rsc_angle_max, args.rsc_angle_max],
                    "ssc_scale_range": [args.ssc_scale_min, args.ssc_scale_max],
                },
            },
        )
        wandb_logger.log_model_summary("PF-CNN + MDA-DMC", n_params)
        print("   W&B logging enabled")

    # Train
    print("\n4. Training with MDA-DMC...")
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
        title="PF-CNN + MDA-DMC Training",
    )
    fig.savefig(args.results_dir / "mda_dmc_training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Evaluate on test set
    print("\n5. Evaluating on test set...")

    # Load best model
    trainer.load_checkpoint(args.checkpoint_dir / "best_model.pt")

    results = evaluate_model(model, loaders["test"], device=config.device)
    print(f"   Test Accuracy: {results['accuracy']:.4f}")

    # SNR sweep
    print("\n6. Computing accuracy vs SNR...")
    snr_values, accuracies = evaluate_snr_sweep(model, loaders["test"], device=config.device)

    # Store results for comparison
    models_acc = {"PF-CNN + MDA-DMC": accuracies}

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
        modulation_classes,
        title="PF-CNN + MDA-DMC Confusion Matrix (All SNRs)",
        ax=ax,
    )
    fig.savefig(args.results_dir / "mda_dmc_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Compare with baseline if requested
    if args.compare_baseline:
        baseline_path = Path("checkpoints/baseline/best_model.pt")
        if baseline_path.exists():
            print("\n8. Comparing with baseline model...")
            baseline_model = create_pfcnn(num_classes=len(modulation_classes), variant="default")
            checkpoint = torch.load(baseline_path, map_location=config.device)
            baseline_model.load_state_dict(checkpoint["model_state_dict"])
            baseline_model.to(config.device)
            baseline_model.eval()

            _, baseline_acc = evaluate_snr_sweep(baseline_model, loaders["test"], device=config.device)
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
    fig.savefig(args.results_dir / "mda_dmc_accuracy_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save results
    results_data = {
        "model": "PF-CNN + MDA-DMC",
        "augmentation": {
            "agn": not args.no_agn,
            "rsc": not args.no_rsc,
            "ssc": not args.no_ssc,
            "prob": args.aug_prob,
            "agn_snr_range": [args.agn_snr_min, args.agn_snr_max],
            "rsc_angle_range": [-args.rsc_angle_max, args.rsc_angle_max],
            "ssc_scale_range": [args.ssc_scale_min, args.ssc_scale_max],
        },
        "best_val_acc": history.best_val_acc,
        "test_acc": results["accuracy"],
        "snr_accuracies": {str(s): float(a) for s, a in zip(snr_values, accuracies)},
    }

    with open(args.results_dir / "mda_dmc_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Log final metrics to W&B
    if wandb_logger is not None:
        wandb_logger.log_snr_accuracy(snr_values, accuracies, "mda_dmc")
        wandb_logger.log_confusion_matrix(
            results["targets"],
            results["predictions"],
            modulation_classes,
            title="MDA-DMC Confusion Matrix",
        )
        wandb_logger.log_image(
            "training_history",
            str(args.results_dir / "mda_dmc_training_history.png"),
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