"""Domain Shift Demo - Observe accuracy collapse under impairments."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from robust_amc.data import RadioMLDataset, PowerNormalize, Compose
from robust_amc.data.transforms import ToTensor
from robust_amc.data.radioml_loader import MODULATION_CLASSES, SNR_LEVELS
from robust_amc.data.channels import RayleighFading, RicianFading
from robust_amc.data.impairments import CarrierFrequencyOffset, IQImbalance, DCOffset
from robust_amc.evaluation.metrics import evaluate_model, accuracy_by_snr

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_dataset, load_model, get_device

st.set_page_config(page_title="Domain Shift Demo", page_icon="📉", layout="wide")

st.title("Domain Shift Demo")
st.markdown("""
**Domain shift** occurs when the test data differs from training data distribution.
This demo shows how a classifier trained on clean signals degrades under real-world impairments.
""")

# Load data and model
dataset = load_dataset()
model = load_model()

if dataset is None:
    st.error("Dataset not found. Please download RadioML2016.10a.")
    st.stop()

if model is None:
    st.error("Model not found. Please train the baseline model first.")
    st.stop()


def evaluate_with_transform(model, data, labels, snrs, transform, batch_size=256):
    """Evaluate model with a specific transform applied."""
    dataset = RadioMLDataset(data, labels, snrs, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    results = evaluate_model(model, loader, get_device())
    return results["accuracy"]


# Get test data
test_data, test_labels, test_snrs = dataset["test"]

# Sidebar options
st.sidebar.header("Demo Options")
demo_type = st.sidebar.selectbox(
    "Select Demo",
    [
        "CFO Sweep",
        "I/Q Imbalance Sweep",
        "DC Offset Sweep",
        "Fading Comparison",
        "Combined Impairments",
    ],
)

n_points = st.sidebar.slider("Number of sweep points", 5, 15, 9)

# Subsample for faster demo
subsample = st.sidebar.checkbox("Use subset (faster)", value=True)
if subsample:
    n_subset = st.sidebar.slider("Subset size", 1000, 10000, 5000, 1000)
    indices = np.random.choice(len(test_labels), size=n_subset, replace=False)
    demo_data = test_data[indices]
    demo_labels = test_labels[indices]
    demo_snrs = test_snrs[indices]
else:
    demo_data = test_data
    demo_labels = test_labels
    demo_snrs = test_snrs

st.sidebar.metric("Samples used", len(demo_labels))


class ImpairmentTransform:
    """Transform that applies impairment then normalization."""

    def __init__(self, impairment):
        self.impairment = impairment
        self.normalize = PowerNormalize()
        self.to_tensor = ToTensor()

    def __call__(self, x):
        x = self.impairment(x)
        x = self.normalize(x)
        x = self.to_tensor(x)
        return x


# Run demo
if st.button("Run Demo", type="primary"):

    if demo_type == "CFO Sweep":
        st.subheader("Carrier Frequency Offset Sweep")
        st.markdown("""
        This sweep shows how accuracy degrades as CFO increases.
        The baseline model was trained on clean signals without CFO.
        """)

        cfo_values = np.linspace(0, 5000, n_points)
        accuracies = []

        progress = st.progress(0)
        status = st.empty()

        for i, cfo in enumerate(cfo_values):
            status.text(f"Evaluating CFO = {cfo:.0f} Hz...")
            impairment = CarrierFrequencyOffset(delta_f=cfo, sample_rate=1e6)
            transform = ImpairmentTransform(impairment)
            acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
            accuracies.append(acc)
            progress.progress((i + 1) / len(cfo_values))

        status.text("Complete!")

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cfo_values, accuracies, "b-o", linewidth=2, markersize=8)
        ax.axhline(y=accuracies[0], color="r", linestyle="--", label=f"Baseline: {accuracies[0]:.1%}")
        ax.fill_between(cfo_values, accuracies, accuracies[0], alpha=0.2, color="red")
        ax.set_xlabel("CFO (Hz)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy vs Carrier Frequency Offset", fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        # Summary
        st.markdown(f"""
        **Results:**
        - Baseline accuracy (CFO=0): **{accuracies[0]:.1%}**
        - Accuracy at CFO=5000Hz: **{accuracies[-1]:.1%}**
        - Degradation: **{(accuracies[0] - accuracies[-1]):.1%}** absolute
        """)

    elif demo_type == "I/Q Imbalance Sweep":
        st.subheader("I/Q Imbalance Sweep")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Amplitude Imbalance**")
            amp_values = np.linspace(0, 3, n_points)
            amp_accuracies = []

            progress = st.progress(0)

            for i, amp in enumerate(amp_values):
                impairment = IQImbalance(amplitude_imbalance_db=amp, phase_imbalance_deg=0)
                transform = ImpairmentTransform(impairment)
                acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
                amp_accuracies.append(acc)
                progress.progress((i + 1) / len(amp_values))

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(amp_values, amp_accuracies, "b-o", linewidth=2)
            ax.axhline(y=amp_accuracies[0], color="r", linestyle="--")
            ax.set_xlabel("Amplitude Imbalance (dB)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Amplitude Imbalance")
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("**Phase Imbalance**")
            phase_values = np.linspace(0, 15, n_points)
            phase_accuracies = []

            progress = st.progress(0)

            for i, phase in enumerate(phase_values):
                impairment = IQImbalance(amplitude_imbalance_db=0, phase_imbalance_deg=phase)
                transform = ImpairmentTransform(impairment)
                acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
                phase_accuracies.append(acc)
                progress.progress((i + 1) / len(phase_values))

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(phase_values, phase_accuracies, "g-o", linewidth=2)
            ax.axhline(y=phase_accuracies[0], color="r", linestyle="--")
            ax.set_xlabel("Phase Imbalance (degrees)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Phase Imbalance")
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(f"""
        **Results:**
        - Amplitude imbalance @ 3dB: **{amp_accuracies[-1]:.1%}** (Δ = {amp_accuracies[0] - amp_accuracies[-1]:.1%})
        - Phase imbalance @ 15°: **{phase_accuracies[-1]:.1%}** (Δ = {phase_accuracies[0] - phase_accuracies[-1]:.1%})
        """)

    elif demo_type == "DC Offset Sweep":
        st.subheader("DC Offset Sweep")

        dc_values = np.linspace(0, 0.3, n_points)
        accuracies = []

        progress = st.progress(0)
        status = st.empty()

        for i, dc in enumerate(dc_values):
            status.text(f"Evaluating DC offset = {dc:.2f}...")
            impairment = DCOffset(dc_i=dc, dc_q=dc, relative=True)
            transform = ImpairmentTransform(impairment)
            acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
            accuracies.append(acc)
            progress.progress((i + 1) / len(dc_values))

        status.text("Complete!")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dc_values, accuracies, "b-o", linewidth=2, markersize=8)
        ax.axhline(y=accuracies[0], color="r", linestyle="--", label=f"Baseline: {accuracies[0]:.1%}")
        ax.fill_between(dc_values, accuracies, accuracies[0], alpha=0.2, color="red")
        ax.set_xlabel("Relative DC Offset", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy vs DC Offset", fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    elif demo_type == "Fading Comparison":
        st.subheader("Fading Channel Comparison")
        st.markdown("""
        Compare accuracy under different fading conditions:
        - **AWGN**: No fading (baseline)
        - **Rayleigh**: Severe fading (no line-of-sight)
        - **Rician**: Moderate fading (with line-of-sight, various K-factors)
        """)

        # Baseline (no fading)
        baseline_transform = Compose([PowerNormalize(), ToTensor()])
        baseline_acc = evaluate_with_transform(
            model, demo_data, demo_labels, demo_snrs, baseline_transform
        )

        # Rayleigh fading (average over realizations)
        st.text("Evaluating Rayleigh fading...")
        rayleigh_accs = []
        for seed in range(5):
            fading = RayleighFading(seed=seed)
            transform = ImpairmentTransform(fading)
            acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
            rayleigh_accs.append(acc)
        rayleigh_acc = np.mean(rayleigh_accs)
        rayleigh_std = np.std(rayleigh_accs)

        # Rician with different K-factors
        k_factors = [0.5, 1, 2, 5, 10, 20]
        rician_accs = []
        rician_stds = []

        progress = st.progress(0)
        for i, k in enumerate(k_factors):
            k_accs = []
            for seed in range(3):
                fading = RicianFading(k_factor=k, seed=seed)
                transform = ImpairmentTransform(fading)
                acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
                k_accs.append(acc)
            rician_accs.append(np.mean(k_accs))
            rician_stds.append(np.std(k_accs))
            progress.progress((i + 1) / len(k_factors))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Baseline
        ax.axhline(y=baseline_acc, color="green", linestyle="-", linewidth=2, label=f"AWGN (Baseline): {baseline_acc:.1%}")

        # Rayleigh
        ax.axhline(y=rayleigh_acc, color="red", linestyle="--", linewidth=2, label=f"Rayleigh: {rayleigh_acc:.1%}")
        ax.fill_between([k_factors[0], k_factors[-1]], rayleigh_acc - rayleigh_std, rayleigh_acc + rayleigh_std, alpha=0.2, color="red")

        # Rician
        ax.errorbar(k_factors, rician_accs, yerr=rician_stds, fmt="b-o", linewidth=2, markersize=8, label="Rician")

        ax.set_xlabel("Rician K-factor", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy Under Different Fading Conditions", fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"""
        **Results:**
        - AWGN (no fading): **{baseline_acc:.1%}**
        - Rayleigh fading: **{rayleigh_acc:.1%}** (Δ = {baseline_acc - rayleigh_acc:.1%})
        - Rician K=5: **{rician_accs[3]:.1%}**
        - Rician K=20: **{rician_accs[-1]:.1%}**

        **Insight:** Higher K-factor means stronger line-of-sight component,
        resulting in less severe fading and better classification accuracy.
        """)

    elif demo_type == "Combined Impairments":
        st.subheader("Combined Impairments")
        st.markdown("""
        Real-world signals typically experience multiple impairments simultaneously.
        This demo shows the compound effect.
        """)

        # Define impairment levels
        levels = [
            {"name": "Clean", "cfo": 0, "iq_amp": 0, "iq_phase": 0, "dc": 0},
            {"name": "Mild", "cfo": 500, "iq_amp": 0.5, "iq_phase": 2, "dc": 0.05},
            {"name": "Moderate", "cfo": 1000, "iq_amp": 1.0, "iq_phase": 5, "dc": 0.1},
            {"name": "Severe", "cfo": 2000, "iq_amp": 2.0, "iq_phase": 10, "dc": 0.2},
            {"name": "Extreme", "cfo": 3000, "iq_amp": 3.0, "iq_phase": 15, "dc": 0.3},
        ]

        accuracies = []
        progress = st.progress(0)

        for i, level in enumerate(levels):
            st.text(f"Evaluating {level['name']} impairments...")

            class CombinedTransform:
                def __init__(self, cfo, iq_amp, iq_phase, dc):
                    self.cfo = CarrierFrequencyOffset(delta_f=cfo, sample_rate=1e6)
                    self.iq = IQImbalance(amplitude_imbalance_db=iq_amp, phase_imbalance_deg=iq_phase)
                    self.dc = DCOffset(dc_i=dc, dc_q=dc, relative=True)
                    self.norm = PowerNormalize()
                    self.to_tensor = ToTensor()

                def __call__(self, x):
                    x = self.cfo(x)
                    x = self.iq(x)
                    x = self.dc(x)
                    x = self.norm(x)
                    x = self.to_tensor(x)
                    return x

            transform = CombinedTransform(
                level["cfo"], level["iq_amp"], level["iq_phase"], level["dc"]
            )
            acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
            accuracies.append(acc)
            progress.progress((i + 1) / len(levels))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [l["name"] for l in levels]
        colors = ["green", "yellowgreen", "yellow", "orange", "red"]

        bars = ax.bar(names, accuracies, color=colors, edgecolor="black", alpha=0.8)

        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{acc:.1%}",
                ha="center",
                fontsize=12,
            )

        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy Under Combined Impairments", fontsize=14)
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.3, axis="y")

        st.pyplot(fig)
        plt.close(fig)

        # Show impairment levels
        st.markdown("**Impairment Levels:**")
        cols = st.columns(5)
        for col, level, acc in zip(cols, levels, accuracies):
            with col:
                st.markdown(f"**{level['name']}**")
                st.markdown(f"CFO: {level['cfo']} Hz")
                st.markdown(f"IQ: {level['iq_amp']}dB / {level['iq_phase']}°")
                st.markdown(f"DC: {level['dc']}")
                st.markdown(f"→ **{acc:.1%}**")

        st.markdown(f"""
        ---
        **Key Insight:** The baseline model experiences severe accuracy degradation
        under realistic impairment conditions. From **{accuracies[0]:.1%}** (clean)
        to **{accuracies[-1]:.1%}** (extreme), a drop of **{accuracies[0] - accuracies[-1]:.1%}** absolute.

        This motivates the need for robust training techniques like:
        - **MDA-DMC**: Multi-domain data augmentation
        - **CLSR-AMC**: Contrastive learning with self-reconstruction
        """)

else:
    st.info("Select a demo type and click 'Run Demo' to see domain shift effects.")

    st.markdown("""
    ### Available Demos

    1. **CFO Sweep**: See how carrier frequency offset degrades accuracy
    2. **I/Q Imbalance Sweep**: Observe effects of amplitude and phase imbalance
    3. **DC Offset Sweep**: Measure sensitivity to DC offset
    4. **Fading Comparison**: Compare AWGN, Rayleigh, and Rician fading
    5. **Combined Impairments**: Realistic scenario with multiple impairments

    ### Why Domain Shift Matters

    Deep learning models learn patterns from training data. When test conditions
    differ (domain shift), performance can degrade significantly. In RF applications,
    domain shift occurs due to:

    - Hardware variations between devices
    - Environmental changes (multipath, fading)
    - Receiver imperfections (CFO, I/Q imbalance)
    - Adversarial conditions

    The next phases of this project will implement techniques to make the
    classifier robust to these conditions.
    """)