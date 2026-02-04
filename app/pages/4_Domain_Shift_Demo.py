"""Domain Shift Demo - Observe accuracy collapse under impairments."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from robust_amc.data import OVERLAPPING_CLASSES, Compose, PowerNormalize, RadioMLDataset
from robust_amc.data.channels import RayleighFading, RicianFading
from robust_amc.data.impairments import CarrierFrequencyOffset, DCOffset, IQImbalance
from robust_amc.data.radioml2018_loader import (
    CLASS_NAME_MAPPING_2018_TO_2016,
    MODULATION_CLASSES_2018,
)
from robust_amc.data.radioml_loader import MODULATION_CLASSES
from robust_amc.data.transforms import ToTensor
from robust_amc.evaluation.metrics import evaluate_model
from robust_amc.utils import get_device

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    DATA_PATH_2018,
    get_available_datasets,
    get_available_models,
    get_dataset_path,
    get_dataset_version,
    load_dataset,
    load_model_by_name,
)

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
Compare how different models handle domain shift.
""")

# Load data
dataset = load_dataset()

if dataset is None:
    st.error("Dataset not found.")
    st.stop()

# Check available models
available_models = get_available_models()

if len(available_models) == 0:
    st.error("No trained models found. Please train at least one model first.")
    st.info("""
    Train models with:
    ```bash
    uv run python scripts/train_baseline.py
    uv run python scripts/train_mda_dmc.py
    uv run python scripts/train_clsr_amc.py
    ```
    """)
    st.stop()

# Model selector in sidebar
st.sidebar.header("Model")
selected_model_name = st.sidebar.selectbox(
    "Select model",
    available_models,
    help="Compare how different models handle domain shift",
)

model = load_model_by_name(selected_model_name)

if model is None:
    st.error(f"Failed to load model: {selected_model_name}")
    st.stop()

st.sidebar.caption(f"Using: {selected_model_name}")


def evaluate_with_transform(model, data, labels, snrs, transform, batch_size=256):
    """Evaluate model with a specific transform."""
    ds = RadioMLDataset(data, labels, snrs, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    results = evaluate_model(model, loader, get_device())
    return results["accuracy"]


# Get test data
test_data, test_labels, test_snrs = dataset["test"]

# Demo type selection
st.sidebar.header("Demo Type")

# Check if 2018 dataset is available for cross-dataset demo
has_2018 = "RadioML2018.01a" in get_available_datasets()

demo_options = ["CFO Sweep", "I/Q Imbalance", "Fading Channels", "Combined"]
if has_2018:
    demo_options.insert(0, "Cross-Dataset (2016→2018)")

demo_type = st.sidebar.selectbox(
    "Demo",
    demo_options,
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

    if demo_type == "Cross-Dataset (2016→2018)":
        st.subheader("Cross-Dataset Domain Shift")
        st.markdown("""
        Testing a model **trained on RadioML2016** against **RadioML2018** test data.
        This demonstrates real-world domain shift from different data collection setups.

        Only the **8 overlapping classes** are used: BPSK, QPSK, 8PSK, QAM16/16QAM, QAM64/64QAM,
        GFSK/GMSK, WBFM/FM, AM-DSB/AM-DSB-SC.
        """)

        # Load 2018 dataset
        with st.spinner("Loading RadioML2018 dataset..."):
            dataset_2018 = load_dataset(DATA_PATH_2018, "2018")

        if dataset_2018 is None:
            st.error("Failed to load RadioML2018 dataset.")
            st.stop()

        # Get 2018 test data
        test_2018_data, test_2018_labels, test_2018_snrs = dataset_2018["test"]
        class_names_2018 = dataset_2018["class_names"]
        snr_levels_2018 = dataset_2018["snr_levels"]

        # Build mapping from 2018 class indices to 2016 class indices (overlapping only)
        # 2018 class name -> 2016 class name -> 2016 index
        class_2018_to_2016_idx = {}
        for cls_2018, cls_2016 in CLASS_NAME_MAPPING_2018_TO_2016.items():
            if cls_2018 in class_names_2018 and cls_2016 in MODULATION_CLASSES:
                idx_2018 = class_names_2018.index(cls_2018)
                idx_2016 = MODULATION_CLASSES.index(cls_2016)
                class_2018_to_2016_idx[idx_2018] = idx_2016

        # Filter 2018 data to overlapping classes only
        overlapping_mask = np.isin(test_2018_labels, list(class_2018_to_2016_idx.keys()))
        filtered_data = test_2018_data[overlapping_mask]
        filtered_labels_2018 = test_2018_labels[overlapping_mask]
        filtered_snrs = test_2018_snrs[overlapping_mask]

        # Remap labels to 2016 indices
        filtered_labels = np.array([class_2018_to_2016_idx[l] for l in filtered_labels_2018])

        st.info(f"Using {len(filtered_labels):,} samples from 2018 dataset ({len(class_2018_to_2016_idx)} overlapping classes)")

        # Evaluate on 2016 test data (baseline)
        st.text("Evaluating on RadioML2016 test data (baseline)...")

        # Filter 2016 data to overlapping classes
        overlapping_2016_indices = [MODULATION_CLASSES.index(c) for c in OVERLAPPING_CLASSES if c in MODULATION_CLASSES]
        mask_2016 = np.isin(demo_labels, overlapping_2016_indices)
        baseline_data = demo_data[mask_2016]
        baseline_labels = demo_labels[mask_2016]
        baseline_snrs = demo_snrs[mask_2016]

        baseline_transform = Compose([PowerNormalize(), ToTensor()])
        baseline_acc = evaluate_with_transform(
            model, baseline_data, baseline_labels, baseline_snrs, baseline_transform
        )

        # Evaluate on 2018 test data
        st.text("Evaluating on RadioML2018 test data...")
        cross_acc = evaluate_with_transform(
            model, filtered_data, filtered_labels, filtered_snrs, baseline_transform
        )

        # Per-SNR analysis
        st.text("Computing per-SNR accuracy...")
        snr_2016_accs = []
        snr_2018_accs = []
        common_snrs = [s for s in snr_levels_2018 if s in dataset["snr_levels"]]

        progress = st.progress(0)
        for i, snr_val in enumerate(common_snrs):
            # 2016
            mask = baseline_snrs == snr_val
            if mask.sum() > 0:
                acc = evaluate_with_transform(
                    model, baseline_data[mask], baseline_labels[mask],
                    baseline_snrs[mask], baseline_transform
                )
                snr_2016_accs.append(acc)
            else:
                snr_2016_accs.append(np.nan)

            # 2018
            mask = filtered_snrs == snr_val
            if mask.sum() > 0:
                acc = evaluate_with_transform(
                    model, filtered_data[mask], filtered_labels[mask],
                    filtered_snrs[mask], baseline_transform
                )
                snr_2018_accs.append(acc)
            else:
                snr_2018_accs.append(np.nan)

            progress.progress((i + 1) / len(common_snrs))

        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart - overall accuracy
        bars = ax1.bar(
            ["2016 (Same Domain)", "2018 (Cross-Domain)"],
            [baseline_acc, cross_acc],
            color=["#22c55e", "#dc2626"],
            alpha=0.85,
            width=0.5,
        )
        for bar, acc in zip(bars, [baseline_acc, cross_acc]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{acc:.1%}", ha="center", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim([0, 1.15])
        ax1.set_title("Overall Accuracy (Overlapping Classes)")
        ax1.grid(True, alpha=0.2, axis="y")

        # Line chart - accuracy vs SNR
        ax2.plot(common_snrs, snr_2016_accs, "o-", label="2016 (Same Domain)",
                color="#22c55e", linewidth=2, markersize=6)
        ax2.plot(common_snrs, snr_2018_accs, "s--", label="2018 (Cross-Domain)",
                color="#dc2626", linewidth=2, markersize=6)
        ax2.set_xlabel("SNR (dB)")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs SNR")
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Same Domain (2016→2016)", f"{baseline_acc:.1%}")
        col2.metric("Cross-Domain (2016→2018)", f"{cross_acc:.1%}")
        drop = baseline_acc - cross_acc
        col3.metric("Domain Shift Drop", f"-{drop:.1%}", delta_color="inverse")

        # Class-wise breakdown
        st.markdown("### Per-Class Analysis")
        class_accs_2016 = []
        class_accs_2018 = []
        class_names_display = []

        for idx_2018, idx_2016 in class_2018_to_2016_idx.items():
            cls_name = MODULATION_CLASSES[idx_2016]
            class_names_display.append(cls_name)

            # 2016 accuracy for this class
            mask = baseline_labels == idx_2016
            if mask.sum() > 0:
                acc = evaluate_with_transform(
                    model, baseline_data[mask], baseline_labels[mask],
                    baseline_snrs[mask], baseline_transform
                )
                class_accs_2016.append(acc)
            else:
                class_accs_2016.append(0)

            # 2018 accuracy for this class
            mask = filtered_labels == idx_2016
            if mask.sum() > 0:
                acc = evaluate_with_transform(
                    model, filtered_data[mask], filtered_labels[mask],
                    filtered_snrs[mask], baseline_transform
                )
                class_accs_2018.append(acc)
            else:
                class_accs_2018.append(0)

        # Plot class-wise comparison
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(class_names_display))
        width = 0.35

        bars1 = ax.bar(x - width/2, class_accs_2016, width, label="2016", color="#22c55e", alpha=0.85)
        bars2 = ax.bar(x + width/2, class_accs_2018, width, label="2018", color="#dc2626", alpha=0.85)

        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Modulation Class")
        ax.set_title("Per-Class Accuracy: Same Domain vs Cross-Domain")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names_display, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.warning("""
        **Key Insight:** The accuracy drop on 2018 data shows **real domain shift**
        from different hardware, channel conditions, and data collection procedures.
        This is more representative of real-world deployment than synthetic impairments.
        """)

    elif demo_type == "CFO Sweep":
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
        than training data. For RF classifiers, this happens in two main ways:

        **1. Hardware/Channel Impairments:**
        - **Training:** Clean AWGN channel, ideal hardware
        - **Real world:** Fading, CFO, I/Q imbalance, DC offset, etc.

        **2. Cross-Dataset Shift:**
        - **Training:** RadioML2016 (specific SDR hardware, channel model)
        - **Deployment:** Different hardware, environment, or data collection setup

        The **Cross-Dataset** demo shows real domain shift by testing a model
        trained on RadioML2016 against RadioML2018 data. This is more representative
        of real-world deployment challenges.

        **Solutions:** Robust training techniques (MDA-DMC, CLSR-AMC) that
        expose the model to various conditions during training.
        """)