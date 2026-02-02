"""Signal Explorer - Visualize I/Q signals and constellations."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from robust_amc.data.radioml_loader import MODULATION_CLASSES, SNR_LEVELS

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_dataset, get_samples_for_modulation

st.set_page_config(page_title="Signal Explorer", page_icon="📊", layout="wide")

st.title("Signal Explorer")
st.markdown("Explore I/Q signals and constellation diagrams from the RadioML2016.10a dataset.")

# Load data
dataset = load_dataset()

if dataset is None:
    st.error("Dataset not found. Please download RadioML2016.10a to `data/RML2016.10a_dict.pkl`")
    st.stop()

# Sidebar controls
st.sidebar.header("Signal Selection")

modulation = st.sidebar.selectbox(
    "Modulation Type",
    MODULATION_CLASSES,
    index=MODULATION_CLASSES.index("QPSK"),
)

snr = st.sidebar.select_slider(
    "SNR (dB)",
    options=SNR_LEVELS,
    value=10,
)

n_samples = st.sidebar.slider("Number of samples", 1, 50, 10)

# Use test set for exploration
test_data, test_labels, test_snrs = dataset["test"]

# Get samples
samples = get_samples_for_modulation(
    test_data, test_labels, test_snrs,
    modulation, snr, n_samples
)

if samples is None:
    st.warning(f"No samples found for {modulation} at {snr} dB")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Constellation Diagram")

    # Plot constellation
    fig, ax = plt.subplots(figsize=(6, 6))

    # Combine all samples
    I = samples[:, 0, :].flatten()
    Q = samples[:, 1, :].flatten()

    ax.scatter(I, Q, alpha=0.3, s=5, c="blue")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(f"{modulation} at {snr} dB SNR")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Set axis limits based on data
    max_val = max(np.abs(I).max(), np.abs(Q).max()) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Time-Domain Signal")

    # Select a single sample to show
    sample_idx = st.selectbox("Sample index", range(len(samples)))
    sample = samples[sample_idx]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    t = np.arange(128)

    axes[0].plot(t, sample[0], "b-", linewidth=1)
    axes[0].set_ylabel("In-phase (I)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Sample {sample_idx}")

    axes[1].plot(t, sample[1], "r-", linewidth=1)
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].set_xlabel("Sample")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Show signal statistics
st.subheader("Signal Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    power = np.mean(samples[:, 0, :] ** 2 + samples[:, 1, :] ** 2)
    st.metric("Average Power", f"{power:.4f}")

with col2:
    amplitude = np.sqrt(samples[:, 0, :] ** 2 + samples[:, 1, :] ** 2)
    st.metric("Mean Amplitude", f"{np.mean(amplitude):.4f}")

with col3:
    st.metric("I Range", f"[{samples[:, 0, :].min():.2f}, {samples[:, 0, :].max():.2f}]")

with col4:
    st.metric("Q Range", f"[{samples[:, 1, :].min():.2f}, {samples[:, 1, :].max():.2f}]")

# Constellation comparison across SNRs
st.subheader("Constellation Comparison Across SNRs")

snr_list = [-10, 0, 10, 18]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, snr_val in zip(axes, snr_list):
    samples_snr = get_samples_for_modulation(
        test_data, test_labels, test_snrs,
        modulation, snr_val, 20
    )

    if samples_snr is not None:
        I = samples_snr[:, 0, :].flatten()
        Q = samples_snr[:, 1, :].flatten()
        ax.scatter(I, Q, alpha=0.3, s=3, c="blue")

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_title(f"{snr_val} dB")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")

fig.suptitle(f"{modulation} Constellation at Different SNRs", fontsize=14)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Modulation comparison at fixed SNR
st.subheader("Modulation Comparison")

selected_mods = st.multiselect(
    "Select modulations to compare",
    MODULATION_CLASSES,
    default=["BPSK", "QPSK", "8PSK", "QAM16"],
)

if selected_mods:
    n_mods = len(selected_mods)
    n_cols = min(4, n_mods)
    n_rows = (n_mods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, mod in enumerate(selected_mods):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        samples_mod = get_samples_for_modulation(
            test_data, test_labels, test_snrs,
            mod, snr, 20
        )

        if samples_mod is not None:
            I = samples_mod[:, 0, :].flatten()
            Q = samples_mod[:, 1, :].flatten()
            ax.scatter(I, Q, alpha=0.3, s=3, c="blue")

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
        ax.set_title(mod)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(len(selected_mods), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"Modulation Comparison at {snr} dB SNR", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)