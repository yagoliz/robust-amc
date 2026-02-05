#!/usr/bin/env python3
"""Evaluate trained model across domains (TorchSig-OOD, Panoradio).

This script evaluates a trained model on:
1. TorchSig in-distribution test set
2. TorchSig out-of-distribution (OOD) test set
3. Panoradio real-world HF signals (zero-shot transfer)

Example usage:
    # Evaluate with default configs
    uv run python scripts/evaluate_cross_domain.py --checkpoint checkpoints/pfcnn_torchsig/best_model.pt

    # Evaluate on Panoradio only
    uv run python scripts/evaluate_cross_domain.py --checkpoint model.pt --panoradio-only
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from robust_amc.data import (
    Compose,
    PowerNormalize,
    load_config_from_yaml,
    get_loaders,
)
from robust_amc.data.transforms import ToTensor
from robust_amc.models import create_pfcnn
from robust_amc.evaluation import (
    evaluate_family_model,
    evaluate_ood_gap,
    evaluate_cross_domain,
    plot_accuracy_vs_snr,
    plot_confusion_matrix,
)
from robust_amc.evaluation.cross_dataset import (
    compute_family_confusion_matrix,
    accuracy_by_family,
    full_evaluation_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-domain evaluation")

    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="default",
        choices=["small", "default", "large"],
        help="Model variant (must match checkpoint)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="Number of output classes (families)",
    )

    # Dataset configs
    parser.add_argument(
        "--torchsig-config",
        type=str,
        default="configs/datasets/torchsig_train.yaml",
        help="TorchSig in-distribution config",
    )
    parser.add_argument(
        "--ood-config",
        type=str,
        default="configs/datasets/torchsig_ood.yaml",
        help="TorchSig OOD config",
    )
    parser.add_argument(
        "--panoradio-config",
        type=str,
        default="configs/datasets/panoradio.yaml",
        help="Panoradio config",
    )

    # Evaluation options
    parser.add_argument(
        "--panoradio-only",
        action="store_true",
        help="Only evaluate on Panoradio",
    )
    parser.add_argument(
        "--skip-panoradio",
        action="store_true",
        help="Skip Panoradio evaluation (if data not available)",
    )

    # Output
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for results",
    )

    # Misc
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")

    return parser.parse_args()


def load_model(checkpoint_path: str, variant: str, num_classes: int, device: str):
    """Load model from checkpoint."""
    model = create_pfcnn(variant=variant, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_device(device_str: str) -> str:
    """Get device string."""
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

    device = get_device(args.device)
    print("=" * 60)
    print("Cross-Domain Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    print(f"\nLoading model ({args.variant}, {args.num_classes} classes)...")
    model = load_model(args.checkpoint, args.variant, args.num_classes, device)

    # Create transform
    transform = Compose([PowerNormalize(), ToTensor()])

    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "checkpoint": args.checkpoint,
        "variant": args.variant,
    }

    # Evaluate on TorchSig in-distribution
    if not args.panoradio_only:
        print("\n" + "-" * 40)
        print("TorchSig In-Distribution Evaluation")
        print("-" * 40)

        try:
            config = load_config_from_yaml(args.torchsig_config)
            loaders = get_loaders(config, eval_transform=transform)
            family_names = loaders["family_names"]

            id_results = evaluate_family_model(
                model, loaders["test"], device, family_names
            )

            print(f"Test accuracy: {id_results['accuracy']:.4f}")
            print("\nPer-family accuracy:")
            family_acc = accuracy_by_family(
                id_results["predictions"],
                id_results["targets"],
                family_names,
            )
            for name, acc in family_acc.items():
                print(f"  {name}: {acc:.4f}")

            results["torchsig_id"] = {
                "accuracy": id_results["accuracy"],
                "snr_accuracy": id_results["snr_accuracy"],
                "per_family_accuracy": family_acc,
            }
            results["family_names"] = family_names

        except Exception as e:
            print(f"Error loading TorchSig data: {e}")
            print("Skipping TorchSig evaluation")

    # Evaluate on TorchSig OOD
    if not args.panoradio_only and not args.skip_panoradio:
        print("\n" + "-" * 40)
        print("TorchSig OOD Evaluation")
        print("-" * 40)

        try:
            ood_config = load_config_from_yaml(args.ood_config)
            ood_loaders = get_loaders(ood_config, eval_transform=transform)

            ood_results = evaluate_family_model(
                model, ood_loaders["test"], device, ood_loaders["family_names"]
            )

            print(f"OOD accuracy: {ood_results['accuracy']:.4f}")

            if "torchsig_id" in results:
                ood_gap = results["torchsig_id"]["accuracy"] - ood_results["accuracy"]
                print(f"OOD gap: {ood_gap:.4f}")
            else:
                ood_gap = None

            results["torchsig_ood"] = {
                "accuracy": ood_results["accuracy"],
                "snr_accuracy": ood_results["snr_accuracy"],
                "ood_gap": ood_gap,
            }

        except Exception as e:
            print(f"Error loading TorchSig OOD data: {e}")
            print("Skipping OOD evaluation")

    # Evaluate on Panoradio
    if not args.skip_panoradio:
        print("\n" + "-" * 40)
        print("Panoradio Zero-Shot Evaluation")
        print("-" * 40)

        try:
            panoradio_config = load_config_from_yaml(args.panoradio_config)
            panoradio_loaders = get_loaders(panoradio_config, eval_transform=transform)

            panoradio_results = evaluate_family_model(
                model, panoradio_loaders["test"], device, panoradio_loaders["family_names"]
            )

            print(f"Zero-shot accuracy: {panoradio_results['accuracy']:.4f}")

            print("\nAccuracy by SNR:")
            for snr, acc in sorted(panoradio_results["snr_accuracy"].items()):
                print(f"  {snr:+3.0f} dB: {acc:.4f}")

            if "torchsig_id" in results:
                domain_gap = results["torchsig_id"]["accuracy"] - panoradio_results["accuracy"]
                print(f"\nDomain gap (TorchSig → Panoradio): {domain_gap:.4f}")
            else:
                domain_gap = None

            results["panoradio"] = {
                "accuracy": panoradio_results["accuracy"],
                "snr_accuracy": panoradio_results["snr_accuracy"],
                "domain_gap": domain_gap,
            }

        except FileNotFoundError as e:
            print(f"Panoradio data not found: {e}")
            print("Download from: https://panoradio-sdr.de/radio-signal-classification-dataset/")
        except Exception as e:
            print(f"Error loading Panoradio data: {e}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = results["timestamp"]
    results_path = results_dir / f"cross_domain_eval_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate comparison plot
    print("\nGenerating plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy comparison
    ax = axes[0]
    datasets = []
    accuracies = []

    if "torchsig_id" in results:
        datasets.append("TorchSig\n(ID)")
        accuracies.append(results["torchsig_id"]["accuracy"])

    if "torchsig_ood" in results:
        datasets.append("TorchSig\n(OOD)")
        accuracies.append(results["torchsig_ood"]["accuracy"])

    if "panoradio" in results:
        datasets.append("Panoradio\n(Zero-shot)")
        accuracies.append(results["panoradio"]["accuracy"])

    if datasets:
        colors = ["#2ecc71", "#f39c12", "#e74c3c"][:len(datasets)]
        bars = ax.bar(datasets, accuracies, color=colors, edgecolor="black")
        ax.set_ylabel("Accuracy")
        ax.set_title("Cross-Domain Accuracy Comparison")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (5 classes)")

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{acc:.1%}", ha="center", va="bottom", fontweight="bold")

    # Plot SNR curves
    ax = axes[1]
    if "torchsig_id" in results:
        snr_acc = results["torchsig_id"]["snr_accuracy"]
        snrs = sorted([float(s) for s in snr_acc.keys()])
        accs = [snr_acc[str(int(s)) if s == int(s) else str(s)] for s in snrs]
        ax.plot(snrs, accs, "g-o", label="TorchSig (ID)", linewidth=2)

    if "torchsig_ood" in results:
        snr_acc = results["torchsig_ood"]["snr_accuracy"]
        snrs = sorted([float(s) for s in snr_acc.keys()])
        accs = [snr_acc[str(int(s)) if s == int(s) else str(s)] for s in snrs]
        ax.plot(snrs, accs, "y-s", label="TorchSig (OOD)", linewidth=2)

    if "panoradio" in results:
        snr_acc = results["panoradio"]["snr_accuracy"]
        snrs = sorted([float(s) for s in snr_acc.keys()])
        accs = [snr_acc[str(int(s)) if s == int(s) else str(s)] for s in snrs]
        ax.plot(snrs, accs, "r-^", label="Panoradio", linewidth=2)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(results_dir / f"cross_domain_comparison_{timestamp}.png", dpi=150)
    print(f"  Saved comparison plot")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "torchsig_id" in results:
        print(f"TorchSig (ID):     {results['torchsig_id']['accuracy']:.2%}")

    if "torchsig_ood" in results:
        print(f"TorchSig (OOD):    {results['torchsig_ood']['accuracy']:.2%}")
        if results["torchsig_ood"]["ood_gap"]:
            print(f"  OOD Gap:         {results['torchsig_ood']['ood_gap']:.2%}")

    if "panoradio" in results:
        print(f"Panoradio:         {results['panoradio']['accuracy']:.2%}")
        if results["panoradio"]["domain_gap"]:
            print(f"  Domain Gap:      {results['panoradio']['domain_gap']:.2%}")

    print("\nDone!")


if __name__ == "__main__":
    main()
