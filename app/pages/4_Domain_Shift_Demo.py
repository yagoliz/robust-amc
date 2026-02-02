"""Domain Shift Demo - Observe accuracy collapse under impairments."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from robust_amc.data import RadioMLDataset, PowerNormalize, Compose
from robust_amc.data.transforms import ToTensor
from robust_amc.data.radioml_loader import MODULATION_CLASSES
from robust_amc.data.channels import RayleighFading, RicianFading
from robust_amc.data.impairments import CarrierFrequencyOffset, IQImbalance, DCOffset
from robust_amc.evaluation.metrics import evaluate_model

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_dataset, load_model, get_device

# Configure matplotlib
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

st.set_page_config(page_title="Domain Shift Demo", page_icon="📉", layout="wide")

st.title("Domain Shift Demo")
st.markdown("""
See how classifier accuracy **collapses** when test conditions differ from training.
The baseline model was trained on clean AWGN signals only.
""")

# Load data and model
dataset = load_dataset()
model = load_model()

if dataset is None:
    st.error("Dataset not found.")
    st.stop()

if model is None:
    st.error("Model not found. Train baseline first.")
    st.stop()


def evaluate_with_transform(model, data, labels, snrs, transform, batch_size=256):
    """Evaluate model with a specific transform."""
    ds = RadioMLDataset(data, labels, snrs, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    results = evaluate_model(model, loader, get_device())
    return results["accuracy"]


# Get test data
test_data, test_labels, test_snrs = dataset["test"]

# Sidebar
st.sidebar.header("Options")

demo_type = st.sidebar.selectbox(
    "Demo",
    ["CFO Sweep", "I/Q Imbalance", "Fading Channels", "Combined"],
)

# Subsample for speed
use_subset = st.sidebar.checkbox("Fast mode (5k samples)", value=True)
if use_subset:
    np.random.seed(42)
    idx = np.random.choice(len(test_labels), size=5000, replace=False)
    demo_data = test_data[idx]
    demo_labels = test_labels[idx]
    demo_snrs = test_snrs[idx]
else:
    demo_data = test_data
    demo_labels = test_labels
    demo_snrs = test_snrs

st.sidebar.caption(f"Using {len(demo_labels):,} samples")


class ImpairmentTransform:
    """Apply impairment then normalize."""
    def __init__(self, impairment):
        self.impairment = impairment
        self.normalize = PowerNormalize()
        self.to_tensor = ToTensor()

    def __call__(self, x):
        x = self.impairment(x)
        x = self.normalize(x)
        x = self.to_tensor(x)
        return x


# Run button
if st.button("Run Demo", type="primary"):

    if demo_type == "CFO Sweep":
        st.subheader("Carrier Frequency Offset")
        st.markdown("Watch accuracy drop as CFO increases from 0 to 5000 Hz.")

        cfo_values = np.linspace(0, 5000, 11)
        accuracies = []

        progress = st.progress(0)

        for i, cfo in enumerate(cfo_values):
            impairment = CarrierFrequencyOffset(delta_f=cfo, sample_rate=1e6)
            transform = ImpairmentTransform(impairment)
            acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
            accuracies.append(acc)
            progress.progress((i + 1) / len(cfo_values))

        # Plot
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(cfo_values, accuracies, "o-", linewidth=2.5, markersize=8, color="#2563eb")
        ax.fill_between(cfo_values, accuracies, accuracies[0], alpha=0.15, color="#dc2626")
        ax.axhline(y=accuracies[0], color="#22c55e", linestyle="--", linewidth=2,
                   label=f"Baseline: {accuracies[0]:.0%}")
        ax.set_xlabel("CFO (Hz)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 5000])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary
        drop = accuracies[0] - accuracies[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline (0 Hz)", f"{accuracies[0]:.0%}")
        col2.metric("At 5000 Hz", f"{accuracies[-1]:.0%}")
        col3.metric("Drop", f"-{drop:.0%}", delta_color="inverse")

    elif demo_type == "I/Q Imbalance":
        st.subheader("I/Q Imbalance")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Amplitude imbalance** (0-3 dB)")
            amp_values = np.linspace(0, 3, 7)
            amp_accs = []
            progress = st.progress(0)

            for i, amp in enumerate(amp_values):
                impairment = IQImbalance(amplitude_imbalance_db=amp, phase_imbalance_deg=0)
                transform = ImpairmentTransform(impairment)
                acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
                amp_accs.append(acc)
                progress.progress((i + 1) / len(amp_values))

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(amp_values, amp_accs, "o-", linewidth=2, markersize=7, color="#2563eb")
            ax.axhline(y=amp_accs[0], color="#22c55e", linestyle="--", linewidth=1.5)
            ax.set_xlabel("Amplitude (dB)")
            ax.set_ylabel("Accuracy")
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.metric("Drop at 3 dB", f"-{amp_accs[0] - amp_accs[-1]:.0%}", delta_color="inverse")

        with col2:
            st.markdown("**Phase imbalance** (0-15°)")
            phase_values = np.linspace(0, 15, 7)
            phase_accs = []
            progress = st.progress(0)

            for i, phase in enumerate(phase_values):
                impairment = IQImbalance(amplitude_imbalance_db=0, phase_imbalance_deg=phase)
                transform = ImpairmentTransform(impairment)
                acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
                phase_accs.append(acc)
                progress.progress((i + 1) / len(phase_values))

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(phase_values, phase_accs, "o-", linewidth=2, markersize=7, color="#8b5cf6")
            ax.axhline(y=phase_accs[0], color="#22c55e", linestyle="--", linewidth=1.5)
            ax.set_xlabel("Phase (degrees)")
            ax.set_ylabel("Accuracy")
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.metric("Drop at 15°", f"-{phase_accs[0] - phase_accs[-1]:.0%}", delta_color="inverse")

    elif demo_type == "Fading Channels":
        st.subheader("Fading Channels")
        st.markdown("Compare AWGN (training), Rayleigh (no LOS), and Rician (with LOS).")

        # Baseline
        baseline_transform = Compose([PowerNormalize(), ToTensor()])
        baseline_acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, baseline_transform)

        # Rayleigh (average 3 seeds)
        st.text("Testing Rayleigh fading...")
        rayleigh_accs = []
        for seed in range(3):
            fading = RayleighFading(seed=seed)
            transform = ImpairmentTransform(fading)
            acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
            rayleigh_accs.append(acc)
        rayleigh_acc = np.mean(rayleigh_accs)

        # Rician with different K
        st.text("Testing Rician fading...")
        k_factors = [1, 2, 5, 10, 20]
        rician_accs = []
        progress = st.progress(0)

        for i, k in enumerate(k_factors):
            k_accs = []
            for seed in range(2):
                fading = RicianFading(k_factor=k, seed=seed)
                transform = ImpairmentTransform(fading)
                acc = evaluate_with_transform(model, demo_data, demo_labels, demo_snrs, transform)
                k_accs.append(acc)
            rician_accs.append(np.mean(k_accs))
            progress.progress((i + 1) / len(k_factors))

        # Plot
        fig, ax = plt.subplots(figsize=(9, 5))

        # Baseline
        ax.axhline(y=baseline_acc, color="#22c55e", linestyle="-", linewidth=2.5,
                   label=f"AWGN: {baseline_acc:.0%}")

        # Rayleigh
        ax.axhline(y=rayleigh_acc, color="#dc2626", linestyle="--", linewidth=2,
                   label=f"Rayleigh: {rayleigh_acc:.0%}")

        # Rician
        ax.plot(k_factors, rician_accs, "o-", linewidth=2.5, markersize=8,
                color="#2563eb", label="Rician")

        ax.set_xlabel("Rician K-factor")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 22])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"""
        **Key insight:** Higher K = stronger line-of-sight = less fading = better accuracy.
        Rayleigh (K→0) is the worst case.
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("AWGN", f"{baseline_acc:.0%}")
        col2.metric("Rayleigh", f"{rayleigh_acc:.0%}", f"-{baseline_acc - rayleigh_acc:.0%}", delta_color="inverse")
        col3.metric("Rician K=10", f"{rician_accs[3]:.0%}", f"-{baseline_acc - rician_accs[3]:.0%}", delta_color="inverse")

    elif demo_type == "Combined":
        st.subheader("Combined Impairments")
        st.markdown("Real-world conditions with multiple impairments stacked.")

        levels = [
            {"name": "Clean", "cfo": 0, "iq_amp": 0, "iq_phase": 0, "dc": 0},
            {"name": "Mild", "cfo": 500, "iq_amp": 0.5, "iq_phase": 2, "dc": 0.05},
            {"name": "Moderate", "cfo": 1000, "iq_amp": 1.0, "iq_phase": 5, "dc": 0.1},
            {"name": "Severe", "cfo": 2000, "iq_amp": 2.0, "iq_phase": 10, "dc": 0.2},
        ]

        accuracies = []
        progress = st.progress(0)

        for i, level in enumerate(levels):

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
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [l["name"] for l in levels]
        colors = ["#22c55e", "#84cc16", "#f59e0b", "#dc2626"]

        bars = ax.bar(names, accuracies, color=colors, alpha=0.85, edgecolor="none", width=0.6)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f"{acc:.0%}", ha="center", fontsize=12, fontweight="bold")

        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary table
        st.markdown("**Impairment levels:**")
        table_data = {
            "Level": names,
            "CFO (Hz)": [l["cfo"] for l in levels],
            "I/Q Amp (dB)": [l["iq_amp"] for l in levels],
            "I/Q Phase (°)": [l["iq_phase"] for l in levels],
            "DC Offset": [l["dc"] for l in levels],
            "Accuracy": [f"{a:.0%}" for a in accuracies],
        }
        st.dataframe(table_data, hide_index=True)

        st.warning(f"**Total drop:** {accuracies[0]:.0%} → {accuracies[-1]:.0%} "
                   f"(**-{accuracies[0] - accuracies[-1]:.0%}**)")

else:
    st.info("Select a demo and click **Run Demo**.")

    with st.expander("What is domain shift?"):
        st.markdown("""
        **Domain shift** occurs when test data comes from a different distribution
        than training data. For RF classifiers:

        - **Training:** Clean AWGN channel, ideal hardware
        - **Real world:** Fading, CFO, I/Q imbalance, etc.

        This demo shows how a baseline model **trained only on clean signals**
        fails when faced with realistic impairments.

        The solution: robust training techniques (MDA-DMC, CLSR-AMC) that
        expose the model to various conditions during training.
        """)