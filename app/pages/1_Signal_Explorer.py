"""Signal Explorer - Visualize I/Q signals and constellations."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    load_dataset,
    get_samples_for_family,
    get_available_datasets,
    get_family_names,
)

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
st.markdown("Visualize I/Q signals and constellation diagrams.")

# Dataset selection
available_datasets = get_available_datasets()
if not available_datasets:
    st.error("No datasets available. TorchSig will be generated on first use.")
    st.stop()

st.sidebar.header("Dataset")
selected_dataset = st.sidebar.selectbox("Dataset", available_datasets)

# Load data
dataset = load_dataset(selected_dataset.lower())

if dataset is None:
    st.error(f"Failed to load {selected_dataset}.")
    st.stop()

# Get family names and SNR levels for selected dataset
family_names = dataset["family_names"]
snr_levels = dataset["snr_levels"]

# Use test set if available, otherwise val
if "test" in dataset:
    test_data, test_labels, test_snrs = dataset["test"]
elif "val" in dataset:
    test_data, test_labels, test_snrs = dataset["val"]
else:
    st.error("No test or validation data available")
    st.stop()

# Sidebar controls
st.sidebar.header("Signal Selection")

# Default family selection
default_family_idx = family_names.index("PSK") if "PSK" in family_names else 0

family = st.sidebar.selectbox(
    "Modulation Family",
    family_names,
    index=default_family_idx,
)
family_idx = family_names.index(family)

# Find a good default SNR
snr_list = sorted(set(int(s) for s in test_snrs))
default_snr = 10 if 10 in snr_list else snr_list[len(snr_list) // 2]

snr = st.sidebar.select_slider(
    "SNR (dB)",
    options=snr_list,
    value=default_snr,
)

n_samples = st.sidebar.slider("Samples to display", 1, 20, 5)

# Get samples
samples = get_samples_for_family(
    test_data, test_labels, test_snrs,
    family_idx, snr, n_samples,
)

if samples is None:
    st.warning(f"No samples found for {family} at {snr} dB")
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
    ax.set_title(f"{family} @ {snr} dB")
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
    seq_len = sample.shape[1]
    t = np.arange(seq_len)

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

# Pick 4 representative SNRs from available range
compare_snrs = [s for s in [-10, 0, 10, 20] if s in snr_list]
if len(compare_snrs) < 4:
    step = max(1, len(snr_list) // 4)
    compare_snrs = snr_list[::step][:4]

fig, axes = plt.subplots(1, len(compare_snrs), figsize=(3.5 * len(compare_snrs), 3.5))
if len(compare_snrs) == 1:
    axes = [axes]

for ax, snr_val in zip(axes, compare_snrs):
    samples_snr = get_samples_for_family(
        test_data, test_labels, test_snrs,
        family_idx, snr_val, 20,
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
    if snr_val == compare_snrs[0]:
        ax.set_ylabel("Q")

plt.suptitle(f"{family} at Different SNRs", fontsize=12)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Family comparison
st.subheader("Family Comparison")

compare_families = st.multiselect(
    "Select families",
    family_names,
    default=family_names[:4],
)

if compare_families:
    n_families = len(compare_families)
    n_cols = min(4, n_families)
    n_rows = (n_families + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))

    # Handle single row/col
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, fam in enumerate(compare_families):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        fam_idx = family_names.index(fam)
        samples_fam = get_samples_for_family(
            test_data, test_labels, test_snrs,
            fam_idx, snr, 20,
        )

        if samples_fam is not None:
            I = samples_fam[:, 0, :].flatten()
            Q = samples_fam[:, 1, :].flatten()
            ax.scatter(I, Q, alpha=0.35, s=4, c="#2563eb", edgecolors='none')

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
        ax.set_title(fam)
        ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
        ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # Hide empty subplots
    for idx in range(len(compare_families), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f"Families @ {snr} dB SNR", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
