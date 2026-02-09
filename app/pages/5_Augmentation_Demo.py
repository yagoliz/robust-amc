"""Augmentation Demo - Visualize MDA-DMC augmentation effects."""

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
    normalize_signal,
)

# Import augmentations
from robust_amc.data.augmentations import (
    AdditiveGaussianNoise,
    RotationInSignalConstellation,
    StretchingInSignalConstellation,
    RotationAndStretchingInSignalConstellation,
    TimeShift,
    RandomFlip,
    MDADMCPipeline,
)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

AXIS_LIMIT = 3.0

st.set_page_config(page_title="Augmentation Demo", page_icon="🔄", layout="wide")

st.title("MDA-DMC Augmentation Demo")
st.markdown(
    "Visualize how **Multi-Domain Augmentation for Domain-Mismatch Compensation (MDA-DMC)** "
    "transforms training data to improve model robustness."
)

# Load dataset
available_datasets = get_available_datasets()
if not available_datasets:
    st.error("No datasets available.")
    st.stop()

# Sidebar configuration
st.sidebar.header("Signal Selection")

selected_dataset = st.sidebar.selectbox("Dataset", available_datasets)
dataset = load_dataset(selected_dataset.lower())

if dataset is None:
    st.error(f"Failed to load {selected_dataset}")
    st.stop()

family_names = dataset["family_names"]

# Use test set
if "test" in dataset:
    test_data, test_labels, test_snrs = dataset["test"]
else:
    test_data, test_labels, test_snrs = dataset["val"]

family = st.sidebar.selectbox("Modulation Family", family_names)
family_idx = family_names.index(family)

snr_list = sorted(set(int(s) for s in test_snrs))
if not snr_list:
    st.error("No SNR values found in the dataset.")
    st.stop()
snr = st.sidebar.select_slider(
    "SNR (dB)",
    options=snr_list,
    value=10 if 10 in snr_list else snr_list[len(snr_list) // 2],
)

# Augmentation selection
st.sidebar.header("Augmentation")

augmentation_type = st.sidebar.selectbox(
    "Augmentation Type",
    [
        "AGN (Additive Gaussian Noise)",
        "RSC (Rotation)",
        "SSC (Stretching)",
        "RSSC (Rotation + Stretching)",
        "Time Shift",
        "Random Flip",
        "Full MDA-DMC Pipeline",
    ]
)

# Augmentation parameters
st.sidebar.header("Parameters")

if "AGN" in augmentation_type:
    snr_min = st.sidebar.slider("Min SNR (dB)", -20, 10, -5)
    snr_max = st.sidebar.slider("Max SNR (dB)", 0, 30, 15)
    aug = AdditiveGaussianNoise(snr_range=(snr_min, snr_max), p=1.0)

elif "RSC" in augmentation_type and "RSSC" not in augmentation_type:
    angle_range = st.sidebar.slider("Rotation Range (deg)", 0, 180, 180)
    aug = RotationInSignalConstellation(angle_range=(-angle_range, angle_range), p=1.0)

elif "SSC" in augmentation_type and "RSSC" not in augmentation_type:
    scale_min = st.sidebar.slider("Min Scale", 0.5, 1.0, 0.8, 0.05)
    scale_max = st.sidebar.slider("Max Scale", 1.0, 2.0, 1.2, 0.05)
    aug = StretchingInSignalConstellation(scale_range=(scale_min, scale_max), p=1.0)

elif "RSSC" in augmentation_type:
    angle_range = st.sidebar.slider("Rotation Range (deg)", 0, 180, 180)
    scale_min = st.sidebar.slider("Min Scale", 0.5, 1.0, 0.8, 0.05)
    scale_max = st.sidebar.slider("Max Scale", 1.0, 2.0, 1.2, 0.05)
    aug = RotationAndStretchingInSignalConstellation(
        angle_range=(-angle_range, angle_range),
        scale_range=(scale_min, scale_max),
        p=1.0,
    )

elif "Time Shift" in augmentation_type:
    max_shift = st.sidebar.slider("Max Shift (samples)", 1, 64, 16)
    aug = TimeShift(max_shift=max_shift, p=1.0)

elif "Random Flip" in augmentation_type:
    flip_i = st.sidebar.checkbox("Flip I channel", value=True)
    flip_q = st.sidebar.checkbox("Flip Q channel", value=True)
    aug = RandomFlip(flip_i=flip_i, flip_q=flip_q, p=1.0)

else:  # Full pipeline
    enable_agn = st.sidebar.checkbox("AGN", value=True)
    enable_rsc = st.sidebar.checkbox("RSC", value=True)
    enable_ssc = st.sidebar.checkbox("SSC", value=True)
    enable_time = st.sidebar.checkbox("Time Shift", value=False)
    enable_flip = st.sidebar.checkbox("Random Flip", value=False)
    prob = st.sidebar.slider("Probability", 0.0, 1.0, 0.5, 0.1)

    aug = MDADMCPipeline(
        agn=enable_agn,
        rsc=enable_rsc,
        ssc=enable_ssc,
        time_shift=enable_time,
        random_flip=enable_flip,
        p=prob,
    )

# Get sample
samples = get_samples_for_family(test_data, test_labels, test_snrs, family_idx, snr, 1)

if samples is None:
    st.warning(f"No samples found for {family} at {snr} dB")
    st.stop()

original = samples[0].copy()

# Apply augmentation button or auto-apply
if st.sidebar.button("Apply Augmentation", type="primary"):
    st.session_state["augmented"] = aug(original.copy())
    st.session_state["original"] = original

if "augmented" not in st.session_state:
    st.session_state["augmented"] = aug(original.copy())
    st.session_state["original"] = original

augmented = st.session_state["augmented"]

# Display comparison
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Signal")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(original[0].flatten(), original[1].flatten(), alpha=0.5, s=8, c="#2563eb")
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
    st.subheader("Augmented Signal")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(augmented[0].flatten(), augmented[1].flatten(), alpha=0.5, s=8, c="#dc2626")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(f"{augmentation_type.split()[0]}")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Time domain comparison
st.markdown("---")
st.subheader("Time Domain")

fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharex=True)
seq_len = original.shape[1]
t = np.arange(seq_len)

axes[0, 0].plot(t, original[0], "-", linewidth=1, color="#2563eb")
axes[0, 0].set_ylabel("I (Original)")
axes[0, 0].grid(True, alpha=0.2)

axes[1, 0].plot(t, original[1], "-", linewidth=1, color="#2563eb")
axes[1, 0].set_ylabel("Q (Original)")
axes[1, 0].set_xlabel("Sample")
axes[1, 0].grid(True, alpha=0.2)

axes[0, 1].plot(t, augmented[0], "-", linewidth=1, color="#dc2626")
axes[0, 1].set_ylabel("I (Augmented)")
axes[0, 1].grid(True, alpha=0.2)

axes[1, 1].plot(t, augmented[1], "-", linewidth=1, color="#dc2626")
axes[1, 1].set_ylabel("Q (Augmented)")
axes[1, 1].set_xlabel("Sample")
axes[1, 1].grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Multiple augmentations gallery
st.markdown("---")
st.subheader("Augmentation Gallery")
st.markdown("Multiple applications of the same augmentation (with randomness):")

n_variants = 6
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for i, ax in enumerate(axes.flatten()):
    if i == 0:
        # Show original
        ax.scatter(original[0].flatten(), original[1].flatten(), alpha=0.5, s=4, c="#2563eb")
        ax.set_title("Original", fontweight='bold')
    else:
        # Show augmented variant
        variant = aug(original.copy())
        ax.scatter(variant[0].flatten(), variant[1].flatten(), alpha=0.5, s=4, c="#dc2626")
        ax.set_title(f"Variant {i}")

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

plt.suptitle(f"{augmentation_type} - Random Variations", fontsize=12)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Explanation section
st.markdown("---")
st.subheader("About MDA-DMC Augmentations")

aug_descriptions = {
    "AGN": """
    **Additive Gaussian Noise (AGN)**: Adds random noise at varying SNR levels.
    This helps the model generalize across different noise conditions encountered in real receivers.
    """,
    "RSC": """
    **Rotation in Signal Constellation (RSC)**: Rotates the I/Q signal by a random angle.
    This simulates carrier phase offset and makes the model invariant to absolute phase,
    which varies randomly in real systems.
    """,
    "SSC": """
    **Stretching in Signal Constellation (SSC)**: Scales the amplitude by a random factor.
    This simulates AGC (Automatic Gain Control) variations and amplitude mismatches
    between training and deployment.
    """,
    "RSSC": """
    **Rotation and Stretching (RSSC)**: Combines RSC and SSC for more comprehensive augmentation.
    Applies both rotation and scaling simultaneously for efficient training.
    """,
    "Time": """
    **Time Shift**: Circularly shifts the signal by a random number of samples.
    This helps the model become invariant to timing offset, which is unknown in real systems.
    """,
    "Random": """
    **Random Flip**: Randomly negates I and/or Q channels.
    This exploits the symmetry properties of most modulation schemes.
    """,
    "Full": """
    **Full MDA-DMC Pipeline**: Chains multiple augmentations together.
    Each augmentation is applied with a configurable probability, creating diverse training samples
    that improve robustness to real-world impairments.
    """,
}

for key, desc in aug_descriptions.items():
    if key in augmentation_type or (key == "Full" and "Pipeline" in augmentation_type):
        st.markdown(desc)
        break

st.markdown("""
### Why Augmentation Works

Data augmentation during training forces the model to learn features that are **invariant**
to the augmented transformations. When these transformations match real-world impairments,
the model becomes robust to domain shift.

The MDA-DMC approach specifically targets the **domain mismatch** between:
- Synthetic training data (clean, simulated)
- Real-world test data (impaired by hardware and channel effects)

By training with augmented data that simulates these impairments, the learned features
generalize better to real conditions.
""")