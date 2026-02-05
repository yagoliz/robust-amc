"""Impairment parameter sweep utilities for robustness evaluation."""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from robust_amc.data.channels import RayleighFading, RicianFading
from robust_amc.data.impairments import CarrierFrequencyOffset, DCOffset, IQImbalance
from robust_amc.data.transforms import Compose, PowerNormalize, ToTensor
from robust_amc.evaluation.metrics import accuracy_by_snr, evaluate_model


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for impairment evaluation."""

    def __init__(self, data, labels, snrs, transform=None):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.snrs = snrs.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        snr = self.snrs[idx]

        if self.transform is not None:
            x = self.transform(x)

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        return x, y, snr


def evaluate_with_impairment(
    model: torch.nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    impairment: Callable,
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

    dataset = SimpleDataset(test_data, test_labels, test_snrs, transform=transform)
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
    cfo_values: Optional[np.ndarray] = None,
    cfo_range: tuple = (0, 5000, 11),
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    """Sweep CFO values and measure accuracy degradation.

    Args:
        model: Trained model
        test_data: Test data array
        test_labels: Test labels
        test_snrs: Test SNR values
        sample_rate: Sample rate in Hz
        cfo_values: Explicit CFO values to sweep (overrides cfo_range)
        cfo_range: (start, stop, num_points) for linspace
        device: Device to run on
        verbose: Whether to show progress bar

    Returns:
        Dictionary with CFO values and accuracies
    """
    if cfo_values is None:
        cfo_values = np.linspace(*cfo_range)

    results = {"cfo_hz": cfo_values.tolist(), "accuracy": [], "snr_accuracy": []}

    iterator = tqdm(cfo_values, desc="CFO sweep") if verbose else cfo_values
    for cfo in iterator:
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
    amp_values: Optional[np.ndarray] = None,
    phase_values: Optional[np.ndarray] = None,
    amp_range: tuple = (0, 3, 7),
    phase_range: tuple = (0, 15, 7),
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    """Sweep I/Q imbalance values.

    Args:
        model: Trained model
        test_data: Test data array
        test_labels: Test labels
        test_snrs: Test SNR values
        amp_values: Explicit amplitude values (dB) to sweep
        phase_values: Explicit phase values (degrees) to sweep
        amp_range: (start, stop, num_points) for amplitude sweep
        phase_range: (start, stop, num_points) for phase sweep
        device: Device to run on
        verbose: Whether to show progress bar

    Returns:
        Dictionary with amplitude and phase sweep results
    """
    if amp_values is None:
        amp_values = np.linspace(*amp_range)
    if phase_values is None:
        phase_values = np.linspace(*phase_range)

    # Sweep amplitude imbalance (with zero phase)
    amp_results = {"amplitude_db": amp_values.tolist(), "accuracy": []}
    iterator = tqdm(amp_values, desc="Amplitude sweep") if verbose else amp_values
    for amp in iterator:
        impairment = IQImbalance(amplitude_imbalance_db=amp, phase_imbalance_deg=0)
        eval_result = evaluate_with_impairment(
            model, test_data, test_labels, test_snrs, impairment, device=device
        )
        amp_results["accuracy"].append(eval_result["overall_accuracy"])

    # Sweep phase imbalance (with zero amplitude)
    phase_results = {"phase_deg": phase_values.tolist(), "accuracy": []}
    iterator = tqdm(phase_values, desc="Phase sweep") if verbose else phase_values
    for phase in iterator:
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
    dc_values: Optional[np.ndarray] = None,
    dc_range: tuple = (0, 0.3, 7),
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    """Sweep DC offset values.

    Args:
        model: Trained model
        test_data: Test data array
        test_labels: Test labels
        test_snrs: Test SNR values
        dc_values: Explicit DC offset values to sweep
        dc_range: (start, stop, num_points) for linspace
        device: Device to run on
        verbose: Whether to show progress bar

    Returns:
        Dictionary with DC offset values and accuracies
    """
    if dc_values is None:
        dc_values = np.linspace(*dc_range)

    results = {"dc_offset": dc_values.tolist(), "accuracy": []}

    iterator = tqdm(dc_values, desc="DC offset sweep") if verbose else dc_values
    for dc in iterator:
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
    k_factors: Optional[list] = None,
    n_realizations: int = 5,
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    """Evaluate under different fading conditions.

    Args:
        model: Trained model
        test_data: Test data array
        test_labels: Test labels
        test_snrs: Test SNR values
        k_factors: Rician K-factors to evaluate (default: [0, 1, 2, 5, 10, 20])
        n_realizations: Number of random realizations for averaging
        device: Device to run on
        verbose: Whether to show progress bar

    Returns:
        Dictionary with Rayleigh and Rician fading results
    """
    if k_factors is None:
        k_factors = [0, 1, 2, 5, 10, 20]

    results = {
        "rayleigh": {"accuracy": [], "snr_accuracy": []},
        "rician": {k: {"accuracy": [], "snr_accuracy": []} for k in k_factors},
    }

    # Rayleigh fading (multiple realizations)
    rayleigh_accs = []
    iterator = tqdm(range(n_realizations), desc="Rayleigh") if verbose else range(n_realizations)
    for seed in iterator:
        impairment = RayleighFading(seed=seed)
        eval_result = evaluate_with_impairment(
            model, test_data, test_labels, test_snrs, impairment, device=device
        )
        rayleigh_accs.append(eval_result["overall_accuracy"])
    results["rayleigh"]["accuracy"] = float(np.mean(rayleigh_accs))
    results["rayleigh"]["std"] = float(np.std(rayleigh_accs))

    # Rician fading with different K-factors
    iterator = tqdm(k_factors, desc="Rician K-factor") if verbose else k_factors
    for k in iterator:
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


def plot_impairment_sweep_results(
    results: dict,
    baseline_accuracy: float,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Generate plots for all sweep results.

    Args:
        results: Dictionary with sweep results (cfo, iq, dc, fading)
        baseline_accuracy: Baseline accuracy without impairments
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # CFO sweep
    if "cfo" in results:
        ax = axes[0, 0]
        ax.plot(results["cfo"]["cfo_hz"], results["cfo"]["accuracy"], "b-o", linewidth=2)
        ax.axhline(y=baseline_accuracy, color="r", linestyle="--", label="Baseline")
        ax.set_xlabel("CFO (Hz)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Carrier Frequency Offset")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # I/Q Imbalance
    if "iq" in results:
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
        ax.axhline(y=baseline_accuracy, color="r", linestyle="--", label="Baseline")
        ax.set_xlabel("Imbalance (dB / degrees)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs I/Q Imbalance")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # DC Offset
    if "dc" in results:
        ax = axes[1, 0]
        ax.plot(
            results["dc"]["dc_offset"],
            results["dc"]["accuracy"],
            "b-o",
            linewidth=2,
        )
        ax.axhline(y=baseline_accuracy, color="r", linestyle="--", label="Baseline")
        ax.set_xlabel("Relative DC Offset")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs DC Offset")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Fading
    if "fading" in results:
        ax = axes[1, 1]
        k_factors = sorted([k for k in results["fading"]["rician"].keys() if isinstance(k, (int, float))])
        rician_accs = [results["fading"]["rician"][k]["accuracy"] for k in k_factors]
        rician_stds = [results["fading"]["rician"][k]["std"] for k in k_factors]

        ax.errorbar(k_factors, rician_accs, yerr=rician_stds, fmt="b-o", label="Rician", linewidth=2)
        ax.axhline(
            y=results["fading"]["rayleigh"]["accuracy"],
            color="orange",
            linestyle="-",
            label=f"Rayleigh ({results['fading']['rayleigh']['accuracy']:.2%})",
        )
        ax.axhline(y=baseline_accuracy, color="r", linestyle="--", label="Baseline")
        ax.set_xlabel("Rician K-factor")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Fading Conditions")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig