"""MDA-DMC Augmentation - Visualize training augmentations for robustness."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from robust_amc.data.radioml_loader import MODULATION_CLASSES, SNR_LEVELS
from robust_amc.data.augmentations import (
    AdditiveGaussianNoise,
    RotationInSignalConstellation,
    StretchingInSignalConstellation,
    RotationAndStretchingInSignalConstellation,
    MDADMCPipeline,
)

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    load_dataset,
    get_samples_for_modulation,
    normalize_samples,
)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

AXIS_LIMIT = 3.0

st.set_page_config(page_title="MDA-DMC Augmentation", page_icon="🔄", layout="wide")

st.title("MDA-DMC Augmentation")
st.markdown("""
**Multi-Domain Augmentation for Domain-Mismatch Compensation** helps improve model robustness
by simulating variations during training. Explore how each augmentation transforms the signal.
""")

# Load data
dataset = load_dataset()

if dataset is None:
    st.error("Dataset not found. Please download RadioML2016.10a.")
    st.stop()

# Sidebar - Signal Selection
st.sidebar.header("Signal Selection")

modulation = st.sidebar.selectbox(
    "Modulation Type",
    MODULATION_CLASSES,
    index=MODULATION_CLASSES.index("QPSK"),
)

snr = st.sidebar.select_slider(
    "SNR (dB)",
    options=SNR_LEVELS,
    value=18,  # High SNR to see augmentation effects clearly
)

# Get samples
test_data, test_labels, test_snrs = dataset["test"]
samples = get_samples_for_modulation(
    test_data, test_labels, test_snrs,
    modulation, snr, 20
)

if samples is None:
    st.warning(f"No samples found for {modulation} at {snr} dB")
    st.stop()

# Sidebar - Augmentation selection
st.sidebar.markdown("---")
st.sidebar.header("Augmentation")

aug_type = st.sidebar.selectbox(
    "Augmentation Type",
    ["AGN (Additive Gaussian Noise)",
     "RSC (Rotation)",
     "SSC (Stretching/Scaling)",
     "RSSC (Rotation + Stretching)",
     "MDA-DMC Pipeline (Combined)"],
)

# Augmentation parameters based on type
st.sidebar.markdown("---")
st.sidebar.header("Parameters")

if "AGN" in aug_type:
    snr_min = st.sidebar.slider("Min SNR (dB)", -10.0, 20.0, -5.0, 1.0)
    snr_max = st.sidebar.slider("Max SNR (dB)", -10.0, 20.0, 15.0, 1.0)
    aug = AdditiveGaussianNoise(snr_range=(snr_min, snr_max), p=1.0, seed=42)
    param_str = f"SNR range: [{snr_min:.0f}, {snr_max:.0f}] dB"

elif "RSC" in aug_type:
    angle_max = st.sidebar.slider("Max Angle (deg)", 0.0, 180.0, 180.0, 5.0)
    aug = RotationInSignalConstellation(angle_range=(-angle_max, angle_max), p=1.0, seed=42)
    param_str = f"Angle range: [-{angle_max:.0f}, +{angle_max:.0f}]°"

elif "SSC" in aug_type and "RSSC" not in aug_type:
    scale_min = st.sidebar.slider("Min Scale", 0.5, 1.0, 0.8, 0.05)
    scale_max = st.sidebar.slider("Max Scale", 1.0, 2.0, 1.2, 0.05)
    aug = StretchingInSignalConstellation(scale_range=(scale_min, scale_max), p=1.0, seed=42)
    param_str = f"Scale range: [{scale_min:.2f}, {scale_max:.2f}]"

elif "RSSC" in aug_type:
    angle_max = st.sidebar.slider("Max Angle (deg)", 0.0, 180.0, 180.0, 5.0)
    scale_min = st.sidebar.slider("Min Scale", 0.5, 1.0, 0.8, 0.05)
    scale_max = st.sidebar.slider("Max Scale", 1.0, 2.0, 1.2, 0.05)
    aug = RotationAndStretchingInSignalConstellation(
        angle_range=(-angle_max, angle_max),
        scale_range=(scale_min, scale_max),
        p=1.0,
        seed=42
    )
    param_str = f"Angle: [-{angle_max:.0f}, +{angle_max:.0f}]°, Scale: [{scale_min:.2f}, {scale_max:.2f}]"

else:  # MDA-DMC Pipeline
    aug_prob = st.sidebar.slider("Augmentation Probability", 0.0, 1.0, 0.5, 0.1)
    use_agn = st.sidebar.checkbox("Enable AGN", value=True)
    use_rsc = st.sidebar.checkbox("Enable RSC", value=True)
    use_ssc = st.sidebar.checkbox("Enable SSC", value=True)
    aug = MDADMCPipeline(agn=use_agn, rsc=use_rsc, ssc=use_ssc, p=aug_prob, seed=42)
    enabled = [name for name, val in [("AGN", use_agn), ("RSC", use_rsc), ("SSC", use_ssc)] if val]
    param_str = f"p={aug_prob:.1f}, Enabled: {', '.join(enabled) or 'None'}"

# Apply augmentation to multiple samples
n_augmented = st.sidebar.slider("Number of Augmentations", 1, 10, 5)

# Use a single sample and augment it multiple times
base_sample = samples[0]
augmented_samples = []
for i in range(n_augmented):
    # Create new augmentation with different seed for variety
    if "AGN" in aug_type:
        aug_i = AdditiveGaussianNoise(snr_range=(snr_min, snr_max), p=1.0, seed=42 + i)
    elif "RSC" in aug_type:
        aug_i = RotationInSignalConstellation(angle_range=(-angle_max, angle_max), p=1.0, seed=42 + i)
    elif "SSC" in aug_type and "RSSC" not in aug_type:
        aug_i = StretchingInSignalConstellation(scale_range=(scale_min, scale_max), p=1.0, seed=42 + i)
    elif "RSSC" in aug_type:
        aug_i = RotationAndStretchingInSignalConstellation(
            angle_range=(-angle_max, angle_max),
            scale_range=(scale_min, scale_max),
            p=1.0, seed=42 + i
        )
    else:
        aug_i = MDADMCPipeline(agn=use_agn, rsc=use_rsc, ssc=use_ssc, p=aug_prob, seed=42 + i)

    augmented = aug_i(base_sample.copy())
    augmented_samples.append(augmented)

augmented_samples = np.array(augmented_samples)
augmented_samples_norm = normalize_samples(augmented_samples)

# Visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Signal")

    fig, ax = plt.subplots(figsize=(5, 5))
    I = samples[:, 0, :].flatten()
    Q = samples[:, 1, :].flatten()
    ax.scatter(I, Q, alpha=0.4, s=8, c="#2563eb", edgecolors='none')
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{modulation} @ {snr} dB")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader(f"After {aug_type.split('(')[0].strip()}")

    fig, ax = plt.subplots(figsize=(5, 5))
    I_aug = augmented_samples_norm[:, 0, :].flatten()
    Q_aug = augmented_samples_norm[:, 1, :].flatten()
    ax.scatter(I_aug, Q_aug, alpha=0.4, s=8, c="#16a34a", edgecolors='none')
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"Augmented ({param_str})")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Info about current augmentation
st.markdown("---")
st.info(f"**Active Augmentation:** {aug_type} | **Parameters:** {param_str}")

# Grid of multiple augmentations
st.markdown("---")
st.subheader("Multiple Augmentation Examples")
st.markdown("Each plot shows the same original signal with a different random augmentation applied.")

n_cols = min(5, n_augmented)
n_rows = (n_augmented + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

for i in range(n_augmented):
    row, col = i // n_cols, i % n_cols
    ax = axes[row, col]

    sample = augmented_samples_norm[i]
    ax.scatter(sample[0], sample[1], alpha=0.5, s=5, c="#16a34a", edgecolors='none')
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.set_title(f"Aug {i+1}", fontsize=10)
    ax.tick_params(labelsize=8)

# Hide empty subplots
for i in range(n_augmented, n_rows * n_cols):
    row, col = i // n_cols, i % n_cols
    axes[row, col].axis('off')

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Explanation section
with st.expander("About MDA-DMC Augmentations", expanded=True):
    st.markdown("""
    ### Multi-Domain Augmentation for Domain-Mismatch Compensation

    MDA-DMC is a data augmentation technique designed to improve the robustness of
    deep learning models for Automatic Modulation Classification (AMC) against domain
    shifts caused by hardware impairments and channel variations.

    #### Augmentation Types

    | Augmentation | Description | Effect |
    |-------------|-------------|--------|
    | **AGN** | Additive Gaussian Noise | Simulates varying SNR conditions |
    | **RSC** | Rotation in Signal Constellation | Random phase rotation in I/Q plane |
    | **SSC** | Stretching in Signal Constellation | Random amplitude scaling |
    | **RSSC** | Combined Rotation + Stretching | Applies both RSC and SSC |

    #### How It Works

    During training, each sample has a probability `p` of having each augmentation applied.
    This exposes the model to a wider variety of signal variations, helping it learn
    features that are invariant to:

    - **Phase offsets** (carrier frequency offset, phase noise)
    - **Amplitude variations** (AGC, fading)
    - **Noise conditions** (varying SNR)

    #### Training with MDA-DMC

    To train with MDA-DMC augmentation:
    ```bash
    uv run python scripts/train_mda_dmc.py --aug-prob 0.5
    ```

    You can also customize individual augmentation parameters:
    ```bash
    uv run python scripts/train_mda_dmc.py \\
        --agn-snr-min -5 --agn-snr-max 15 \\
        --rsc-angle-max 180 \\
        --ssc-scale-min 0.8 --ssc-scale-max 1.2
    ```
    """)

# Before/After comparison with specific examples
with st.expander("Effect Demonstration", expanded=False):
    st.markdown("### How Each Augmentation Affects the Signal")

    # Create one example of each augmentation type
    demo_augs = [
        ("Original", None),
        ("AGN (SNR jitter)", AdditiveGaussianNoise(snr_range=(5, 10), p=1.0, seed=42)),
        ("RSC (45° rotation)", RotationInSignalConstellation(angle_range=(45, 45), p=1.0)),
        ("SSC (1.5x scale)", StretchingInSignalConstellation(scale_range=(1.5, 1.5), p=1.0)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for i, (name, aug_demo) in enumerate(demo_augs):
        if aug_demo is None:
            sample_demo = samples[0]
        else:
            sample_demo = aug_demo(samples[0].copy())

        sample_demo_norm = normalize_samples(sample_demo.reshape(1, 2, -1))[0]

        ax = axes[i]
        ax.scatter(sample_demo_norm[0], sample_demo_norm[1], alpha=0.5, s=8,
                   c="#2563eb" if i == 0 else "#16a34a", edgecolors='none')
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
        ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("I")
        if i == 0:
            ax.set_ylabel("Q")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)