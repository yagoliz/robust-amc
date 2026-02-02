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

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

# Fixed axis limits for consistent comparison
AXIS_LIMIT = 3.0

st.set_page_config(page_title="Signal Explorer", page_icon="📊", layout="wide")

st.title("Signal Explorer")
st.markdown("Visualize I/Q signals and constellation diagrams from RadioML2016.10a.")

# Load data
dataset = load_dataset()

if dataset is None:
    st.error("Dataset not found. Please download RadioML2016.10a to `data/RML2016.10a_dict.pkl`")
    st.stop()

# Sidebar controls
st.sidebar.header("Signal Selection")

modulation = st.sidebar.selectbox(
    "Modulation",
    MODULATION_CLASSES,
    index=MODULATION_CLASSES.index("QPSK"),
)

snr = st.sidebar.select_slider(
    "SNR (dB)",
    options=SNR_LEVELS,
    value=10,
)

n_samples = st.sidebar.slider("Samples to display", 5, 50, 20)

# Use test set
test_data, test_labels, test_snrs = dataset["test"]

# Get samples
samples = get_samples_for_modulation(
    test_data, test_labels, test_snrs,
    modulation, snr, n_samples
)

if samples is None:
    st.warning(f"No samples found for {modulation} at {snr} dB")
    st.stop()

# Main view
col1, col2 = st.columns(2)

with col1:
    st.subheader("Constellation")

    fig, ax = plt.subplots(figsize=(5, 5))
    I = samples[:, 0, :].flatten()
    Q = samples[:, 1, :].flatten()

    ax.scatter(I, Q, alpha=0.35, s=6, c="#2563eb", edgecolors='none')
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(f"{modulation} @ {snr} dB")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Time Domain")

    sample_idx = st.selectbox("Sample", range(len(samples)), format_func=lambda x: f"#{x+1}")
    sample = samples[sample_idx]

    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    t = np.arange(128)

    axes[0].plot(t, sample[0], "-", linewidth=1, color="#2563eb")
    axes[0].set_ylabel("I")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t, sample[1], "-", linewidth=1, color="#dc2626")
    axes[1].set_ylabel("Q")
    axes[1].set_xlabel("Sample")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# SNR comparison
st.markdown("---")
st.subheader("SNR Comparison")

snr_list = [-10, 0, 10, 18]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

for ax, snr_val in zip(axes, snr_list):
    samples_snr = get_samples_for_modulation(
        test_data, test_labels, test_snrs,
        modulation, snr_val, 20
    )

    if samples_snr is not None:
        I = samples_snr[:, 0, :].flatten()
        Q = samples_snr[:, 1, :].flatten()
        ax.scatter(I, Q, alpha=0.35, s=4, c="#2563eb", edgecolors='none')

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.set_title(f"{snr_val} dB")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("I")
    if snr_val == snr_list[0]:
        ax.set_ylabel("Q")

plt.suptitle(f"{modulation} at Different SNRs", fontsize=12)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Modulation comparison
st.subheader("Modulation Comparison")

compare_mods = st.multiselect(
    "Select modulations",
    MODULATION_CLASSES,
    default=["BPSK", "QPSK", "8PSK", "QAM16"],
)

if compare_mods:
    n_mods = len(compare_mods)
    n_cols = min(4, n_mods)
    n_rows = (n_mods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))

    # Handle single row/col
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, mod in enumerate(compare_mods):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        samples_mod = get_samples_for_modulation(
            test_data, test_labels, test_snrs,
            mod, snr, 20
        )

        if samples_mod is not None:
            I = samples_mod[:, 0, :].flatten()
            Q = samples_mod[:, 1, :].flatten()
            ax.scatter(I, Q, alpha=0.35, s=4, c="#2563eb", edgecolors='none')

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
        ax.set_title(mod)
        ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
        ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # Hide empty subplots
    for idx in range(len(compare_mods), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f"Modulations @ {snr} dB SNR", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)