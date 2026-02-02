#!/usr/bin/env python3
"""Verify data loading and visualize example constellations."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from robust_amc.data import (
    RadioMLDataset,
    load_radioml2016a,
    get_data_loaders,
    PowerNormalize,
    Compose,
)
from robust_amc.data.radioml_loader import (
    MODULATION_CLASSES,
    SNR_LEVELS,
    get_samples_by_modulation_snr,
    stratified_split,
)
from robust_amc.data.transforms import ToTensor
from robust_amc.evaluation import plot_constellation, plot_constellation_grid


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "RML2016.10a_dict.pkl"
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("RadioML2016.10a Dataset Verification")
    print("=" * 60)

    # Check if dataset exists
    if not data_path.exists():
        print(f"\nDataset not found at {data_path}")
        print("Run: python scripts/download_data.py")
        sys.exit(1)

    # Load raw data
    print("\n1. Loading dataset...")
    data, labels, snrs = load_radioml2016a(data_path)

    print(f"   Data shape: {data.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   SNRs shape: {snrs.shape}")
    print(f"   Data dtype: {data.dtype}")
    print(f"   Data range: [{data.min():.4f}, {data.max():.4f}]")

    # Verify modulation distribution
    print("\n2. Modulation distribution:")
    for idx, mod in enumerate(MODULATION_CLASSES):
        count = (labels == idx).sum()
        print(f"   {mod:8s}: {count:6d} samples")

    # Verify SNR distribution
    print("\n3. SNR distribution:")
    for snr in SNR_LEVELS:
        count = (snrs == snr).sum()
        print(f"   {snr:4d} dB: {count:6d} samples")

    # Test stratified split
    print("\n4. Testing stratified split...")
    splits = stratified_split(data, labels, snrs)
    for name, (d, l, s) in splits.items():
        print(f"   {name:5s}: {len(l):6d} samples ({100*len(l)/len(labels):.1f}%)")

    # Create dataset with transforms
    print("\n5. Testing dataset with transforms...")
    transform = Compose([PowerNormalize(), ToTensor()])
    dataset = RadioMLDataset(data, labels, snrs, transform=transform)

    sample, label, snr = dataset[0]
    print(f"   Sample shape: {sample.shape}")
    print(f"   Sample dtype: {sample.dtype}")
    print(f"   Sample power: {(sample[0]**2 + sample[1]**2).mean():.4f} (should be ~1.0)")

    # Test data loaders
    print("\n6. Testing data loaders...")
    loaders = get_data_loaders(
        data_path,
        batch_size=64,
        train_transform=transform,
        eval_transform=transform,
        num_workers=0,  # Use 0 for testing
    )

    for name, loader in loaders.items():
        batch = next(iter(loader))
        x, y, s = batch
        print(f"   {name:5s} batch: x={tuple(x.shape)}, y={tuple(y.shape)}, snr={tuple(s.shape)}")

    # Visualize constellations
    print("\n7. Generating constellation plots...")

    # Create dataset without normalization for visualization
    raw_dataset = RadioMLDataset(data, labels, snrs)

    # Get samples for each modulation at 10 dB
    snr_for_plot = 10
    samples_dict = {}

    for mod in MODULATION_CLASSES:
        try:
            samples, _ = get_samples_by_modulation_snr(raw_dataset, mod, snr_for_plot, n_samples=100)
            samples_dict[mod] = samples
        except ValueError as e:
            print(f"   Warning: {e}")

    # Plot constellation grid
    fig = plot_constellation_grid(
        samples_dict,
        n_cols=4,
        suptitle=f"Constellation Diagrams at SNR = {snr_for_plot} dB",
    )

    output_path = output_dir / "constellation_grid.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   Saved: {output_path}")
    plt.close(fig)

    # Plot single constellation with different SNRs
    print("\n8. Generating SNR comparison plot...")
    mod_to_show = "QPSK"
    snrs_to_show = [-10, 0, 10, 18]

    fig, axes = plt.subplots(1, len(snrs_to_show), figsize=(4 * len(snrs_to_show), 4))

    for idx, snr in enumerate(snrs_to_show):
        try:
            samples, _ = get_samples_by_modulation_snr(raw_dataset, mod_to_show, snr, n_samples=200)
            plot_constellation(samples, title=f"{mod_to_show} @ {snr} dB", ax=axes[idx])
        except ValueError as e:
            print(f"   Warning: {e}")

    fig.suptitle(f"{mod_to_show} Constellation at Different SNR Levels", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / f"{mod_to_show.lower()}_snr_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   Saved: {output_path}")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
