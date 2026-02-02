#!/usr/bin/env python3
"""Evaluate model robustness under various impairments.

This script demonstrates domain shift by showing how a baseline model's
accuracy collapses under various hardware impairments and channel effects.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from robust_amc.data import (
    RadioMLDataset,
    load_radioml2016a,
    stratified_split,
)
from robust_amc.data.transforms import Compose, PowerNormalize, ToTensor
from robust_amc.data.channels import AWGN, RayleighFading, RicianFading
from robust_amc.data.impairments import (
    CarrierFrequencyOffset,
    IQImbalance,
    DCOffset,
    PhaseNoise,
)
from robust_amc.models import PFCNN
from robust_amc.evaluation.metrics import evaluate_model, accuracy_by_snr


def evaluate_with_impairment(
    model: torch.nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    impairment,
    batch_size: int = 256,
    device: str = "auto",
) -> dict:
    """Evaluate model with a specific impairment applied.

    Args:
        model: Trained model
        test_data: Test data array (N, 2, 128)
        test_labels: Test labels
        test_snrs: Test SNR values
        impairment: Impairment transform to apply
        batch_size: Batch size for evaluation
        device: Device to run on

    Returns:
        Dictionary with accuracy and SNR breakdown
    """
    # Create transform with impairment
    transform = Compose([
        impairment,
        PowerNormalize(),
        ToTensor(),
    ])

    dataset = RadioMLDataset(test_data, test_labels, test_snrs, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = evaluate_model(model, loader, device)
    snr_acc = accuracy_by_snr(results["targets"], results["predictions"], results["snrs"])

    return {
        "overall_accuracy": results["accuracy"],
        "snr_accuracy": snr_acc,
    }


def sweep_cfo(
    model: torch.nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    sample_rate: float = 1e6,
    cfo_range: tuple = (0, 5000, 11),  # start, stop, num_points
    device: str = "auto",
) -> dict:
    """Sweep CFO values and measure accuracy degradation."""
    cfo_values = np.linspace(*cfo_range)
    results = {"cfo_hz": cfo_values.tolist(), "accuracy": [], "snr_accuracy": []}

    print("Sweeping CFO values...")
    for cfo in tqdm(cfo_values, desc="CFO sweep"):
        impairment = CarrierFrequencyOffset(delta_f=cfo, sample_rate=sample_rate)
        eval_result = evaluate_with_impairment(
            model, test_data, test_labels, test_snrs, impairment, device=device
        )
        results["accuracy"].append(eval_result["overall_accuracy"])
        results["snr_accuracy"].append(eval_result["snr_accuracy"])

    return results


def sweep_iq_imbalance(
    model: torch.nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    amp_range: tuple = (0, 3, 7),  # dB
    phase_range: tuple = (0, 15, 7),  # degrees
    device: str = "auto",
) -> dict:
    """Sweep I/Q imbalance values."""
    # Sweep amplitude imbalance (with zero phase)
    amp_values = np.linspace(*amp_range)
    amp_results = {"amplitude_db": amp_values.tolist(), "accuracy": []}

    print("Sweeping amplitude imbalance...")
    for amp in tqdm(amp_values, desc="Amplitude sweep"):
        impairment = IQImbalance(amplitude_imbalance_db=amp, phase_imbalance_deg=0)
        eval_result = evaluate_with_impairment(
            model, test_data, test_labels, test_snrs, impairment, device=device
        )
        amp_results["accuracy"].append(eval_result["overall_accuracy"])

    # Sweep phase imbalance (with zero amplitude)
    phase_values = np.linspace(*phase_range)
    phase_results = {"phase_deg": phase_values.tolist(), "accuracy": []}

    print("Sweeping phase imbalance...")
    for phase in tqdm(phase_values, desc="Phase sweep"):
        impairment = IQImbalance(amplitude_imbalance_db=0, phase_imbalance_deg=phase)
        eval_result = evaluate_with_impairment(
            model, test_data, test_labels, test_snrs, impairment, device=device
        )
        phase_results["accuracy"].append(eval_result["overall_accuracy"])

    return {"amplitude": amp_results, "phase": phase_results}


def sweep_dc_offset(
    model: torch.nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    dc_range: tuple = (0, 0.3, 7),  # relative DC offset
    device: str = "auto",
) -> dict:
    """Sweep DC offset values."""
    dc_values = np.linspace(*dc_range)
    results = {"dc_offset": dc_values.tolist(), "accuracy": []}

    print("Sweeping DC offset...")
    for dc in tqdm(dc_values, desc="DC offset sweep"):
        impairment = DCOffset(dc_i=dc, dc_q=dc, relative=True)
        eval_result = evaluate_with_impairment(
            model, test_data, test_labels, test_snrs, impairment, device=device
        )
        results["accuracy"].append(eval_result["overall_accuracy"])

    return results


def sweep_fading(
    model: torch.nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    k_factors: list = [0, 1, 2, 5, 10, 20],  # Rician K-factors
    n_realizations: int = 5,
    device: str = "auto",
) -> dict:
    """Evaluate under different fading conditions."""
    results = {
        "rayleigh": {"accuracy": [], "snr_accuracy": []},
        "rician": {k: {"accuracy": [], "snr_accuracy": []} for k in k_factors},
    }

    # Rayleigh fading (multiple realizations)
    print("Evaluating Rayleigh fading...")
    rayleigh_accs = []
    for seed in tqdm(range(n_realizations), desc="Rayleigh"):
        impairment = RayleighFading(seed=seed)
        eval_result = evaluate_with_impairment(
            model, test_data, test_labels, test_snrs, impairment, device=device
        )
        rayleigh_accs.append(eval_result["overall_accuracy"])
    results["rayleigh"]["accuracy"] = float(np.mean(rayleigh_accs))
    results["rayleigh"]["std"] = float(np.std(rayleigh_accs))

    # Rician fading with different K-factors
    print("Evaluating Rician fading...")
    for k in tqdm(k_factors, desc="Rician K-factor"):
        k_accs = []
        for seed in range(n_realizations):
            impairment = RicianFading(k_factor=k, seed=seed)
            eval_result = evaluate_with_impairment(
                model, test_data, test_labels, test_snrs, impairment, device=device
            )
            k_accs.append(eval_result["overall_accuracy"])
        results["rician"][k]["accuracy"] = float(np.mean(k_accs))
        results["rician"][k]["std"] = float(np.std(k_accs))

    return results


def plot_results(results: dict, output_dir: Path):
    """Generate plots for all sweep results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # CFO sweep
    ax = axes[0, 0]
    ax.plot(results["cfo"]["cfo_hz"], results["cfo"]["accuracy"], "b-o", linewidth=2)
    ax.axhline(y=results["baseline_accuracy"], color="r", linestyle="--", label="Baseline")
    ax.set_xlabel("CFO (Hz)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Carrier Frequency Offset")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # I/Q Imbalance
    ax = axes[0, 1]
    ax.plot(
        results["iq"]["amplitude"]["amplitude_db"],
        results["iq"]["amplitude"]["accuracy"],
        "b-o",
        label="Amplitude",
        linewidth=2,
    )
    ax.plot(
        results["iq"]["phase"]["phase_deg"],
        results["iq"]["phase"]["accuracy"],
        "g-s",
        label="Phase",
        linewidth=2,
    )
    ax.axhline(y=results["baseline_accuracy"], color="r", linestyle="--", label="Baseline")
    ax.set_xlabel("Imbalance (dB / degrees)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs I/Q Imbalance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # DC Offset
    ax = axes[1, 0]
    ax.plot(
        results["dc"]["dc_offset"],
        results["dc"]["accuracy"],
        "b-o",
        linewidth=2,
    )
    ax.axhline(y=results["baseline_accuracy"], color="r", linestyle="--", label="Baseline")
    ax.set_xlabel("Relative DC Offset")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs DC Offset")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fading
    ax = axes[1, 1]
    k_factors = sorted([k for k in results["fading"]["rician"].keys() if k != "accuracy"])
    rician_accs = [results["fading"]["rician"][k]["accuracy"] for k in k_factors]
    rician_stds = [results["fading"]["rician"][k]["std"] for k in k_factors]

    ax.errorbar(k_factors, rician_accs, yerr=rician_stds, fmt="b-o", label="Rician", linewidth=2)
    ax.axhline(
        y=results["fading"]["rayleigh"]["accuracy"],
        color="orange",
        linestyle="-",
        label=f"Rayleigh ({results['fading']['rayleigh']['accuracy']:.2%})",
    )
    ax.axhline(y=results["baseline_accuracy"], color="r", linestyle="--", label="Baseline")
    ax.set_xlabel("Rician K-factor")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Fading Conditions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "impairment_sweep.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "impairment_sweep.pdf", bbox_inches="tight")
    print(f"Saved plots to {output_dir}")

    # Also create SNR breakdown plot for CFO
    if results["cfo"]["snr_accuracy"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        snr_values = sorted(results["cfo"]["snr_accuracy"][0].keys())

        for i, (cfo, snr_acc) in enumerate(
            zip(results["cfo"]["cfo_hz"][::2], results["cfo"]["snr_accuracy"][::2])
        ):
            accs = [snr_acc[snr] for snr in snr_values]
            ax.plot(snr_values, accs, "-o", label=f"CFO={cfo:.0f}Hz", alpha=0.8)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs SNR for Different CFO Values")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "cfo_snr_breakdown.png", dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model under impairments")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/RML2016.10a_dict.pkl",
        help="Path to RadioML dataset",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pfcnn_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/impairment_sweep",
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1e6,
        help="Sample rate for CFO calculation (Hz)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda, mps)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = PFCNN(num_classes=11)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load test data
    print(f"Loading data from {args.data_path}")
    data, labels, snrs = load_radioml2016a(args.data_path)
    splits = stratified_split(data, labels, snrs)
    test_data, test_labels, test_snrs = splits["test"]

    print(f"Test set size: {len(test_labels)}")

    # Get baseline accuracy (no impairments)
    print("\nEvaluating baseline (no impairments)...")
    baseline_transform = Compose([PowerNormalize(), ToTensor()])
    baseline_dataset = RadioMLDataset(
        test_data, test_labels, test_snrs, transform=baseline_transform
    )
    baseline_loader = DataLoader(baseline_dataset, batch_size=256, shuffle=False, num_workers=0)
    baseline_results = evaluate_model(model, baseline_loader, args.device)
    baseline_accuracy = baseline_results["accuracy"]
    print(f"Baseline accuracy: {baseline_accuracy:.2%}")

    # Run sweeps
    results = {
        "baseline_accuracy": baseline_accuracy,
        "cfo": sweep_cfo(
            model, test_data, test_labels, test_snrs,
            sample_rate=args.sample_rate,
            device=args.device,
        ),
        "iq": sweep_iq_imbalance(
            model, test_data, test_labels, test_snrs,
            device=args.device,
        ),
        "dc": sweep_dc_offset(
            model, test_data, test_labels, test_snrs,
            device=args.device,
        ),
        "fading": sweep_fading(
            model, test_data, test_labels, test_snrs,
            device=args.device,
        ),
    }

    # Save results
    results_path = output_dir / "impairment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    # Generate plots
    plot_results(results, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Domain Shift Impact")
    print("=" * 60)
    print(f"Baseline accuracy: {baseline_accuracy:.2%}")
    print(f"\nCFO impact (at {results['cfo']['cfo_hz'][-1]:.0f} Hz):")
    print(f"  Accuracy: {results['cfo']['accuracy'][-1]:.2%} ({results['cfo']['accuracy'][-1] - baseline_accuracy:+.2%})")
    print(f"\nI/Q amplitude imbalance impact (at {results['iq']['amplitude']['amplitude_db'][-1]:.1f} dB):")
    print(f"  Accuracy: {results['iq']['amplitude']['accuracy'][-1]:.2%} ({results['iq']['amplitude']['accuracy'][-1] - baseline_accuracy:+.2%})")
    print(f"\nI/Q phase imbalance impact (at {results['iq']['phase']['phase_deg'][-1]:.1f} deg):")
    print(f"  Accuracy: {results['iq']['phase']['accuracy'][-1]:.2%} ({results['iq']['phase']['accuracy'][-1] - baseline_accuracy:+.2%})")
    print(f"\nRayleigh fading impact:")
    print(f"  Accuracy: {results['fading']['rayleigh']['accuracy']:.2%} ({results['fading']['rayleigh']['accuracy'] - baseline_accuracy:+.2%})")


if __name__ == "__main__":
    main()