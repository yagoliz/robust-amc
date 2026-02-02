"""Impairment Simulator - Interactive exploration of hardware impairments."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from robust_amc.data.radioml_loader import MODULATION_CLASSES, SNR_LEVELS

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    load_dataset,
    load_model,
    get_samples_for_modulation,
    apply_impairments,
    apply_fading,
    predict_modulation,
    normalize_samples,
)

# Configure matplotlib for cleaner plots
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Fixed axis limit for constellation plots (after normalization signals are roughly in this range)
AXIS_LIMIT = 3.0

st.set_page_config(page_title="Impairment Simulator", page_icon="⚡", layout="wide")

st.title("Impairment Simulator")
st.markdown("Adjust sliders to apply impairments and observe constellation changes in real-time.")

# Load data and model
dataset = load_dataset()
model = load_model()

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
    value=10,
)

# Get test samples
test_data, test_labels, test_snrs = dataset["test"]
samples = get_samples_for_modulation(
    test_data, test_labels, test_snrs,
    modulation, snr, 20
)

if samples is None:
    st.warning(f"No samples found for {modulation} at {snr} dB")
    st.stop()

# Main content - Impairment controls
st.sidebar.markdown("---")
st.sidebar.header("Hardware Impairments")

# CFO
cfo_hz = st.sidebar.slider("CFO (Hz)", 0, 5000, 0, 100)

# I/Q Imbalance
iq_amp_db = st.sidebar.slider("I/Q Amplitude (dB)", 0.0, 3.0, 0.0, 0.1)
iq_phase_deg = st.sidebar.slider("I/Q Phase (deg)", 0.0, 15.0, 0.0, 0.5)

# DC Offset
dc_offset = st.sidebar.slider("DC Offset", 0.0, 0.5, 0.0, 0.02)

# Phase Noise
phase_noise_std = st.sidebar.slider("Phase Noise", 0.0, 0.1, 0.0, 0.005)

# Fading
st.sidebar.markdown("---")
st.sidebar.header("Channel")
fading_type = st.sidebar.selectbox("Fading", ["none", "rayleigh", "rician"])
k_factor = 5.0
if fading_type == "rician":
    k_factor = st.sidebar.slider("Rician K-factor", 0.0, 20.0, 5.0, 1.0)

# Apply impairments to all samples
impaired_samples_raw = []
for sample in samples:
    faded = apply_fading(sample, fading_type, k_factor)
    impaired = apply_impairments(
        faded,
        cfo_hz=cfo_hz,
        iq_amp_db=iq_amp_db,
        iq_phase_deg=iq_phase_deg,
        dc_i=dc_offset,
        dc_q=dc_offset,
        phase_noise_std=phase_noise_std,
    )
    impaired_samples_raw.append(impaired)

impaired_samples_raw = np.array(impaired_samples_raw)

# Normalize for display (keeps visualization on consistent scale)
impaired_samples = normalize_samples(impaired_samples_raw)

# Check if any impairment is active
has_impairments = (cfo_hz > 0 or iq_amp_db > 0 or iq_phase_deg > 0 or
                   dc_offset > 0 or phase_noise_std > 0 or fading_type != "none")

# Visualization - Constellation plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Clean")

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

    # Model prediction on clean signal
    if model is not None:
        pred_clean, probs_clean = predict_modulation(model, samples[0])
        conf = probs_clean[MODULATION_CLASSES.index(pred_clean)]
        if pred_clean == modulation:
            st.success(f"Prediction: **{pred_clean}** ({conf:.0%})")
        else:
            st.error(f"Prediction: **{pred_clean}** ({conf:.0%}) - True: {modulation}")

with col2:
    st.subheader("Impaired" if has_impairments else "Clean (no impairments)")

    fig, ax = plt.subplots(figsize=(5, 5))
    I_imp = impaired_samples[:, 0, :].flatten()
    Q_imp = impaired_samples[:, 1, :].flatten()
    color = "#dc2626" if has_impairments else "#2563eb"
    ax.scatter(I_imp, Q_imp, alpha=0.4, s=8, c=color, edgecolors='none')
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # Simple title
    if has_impairments:
        ax.set_title(f"{modulation} + Impairments")
    else:
        ax.set_title(f"{modulation} @ {snr} dB")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Model prediction on impaired signal
    if model is not None:
        pred_imp, probs_imp = predict_modulation(model, impaired_samples_raw[0])
        conf = probs_imp[MODULATION_CLASSES.index(pred_imp)]
        if pred_imp == modulation:
            st.success(f"Prediction: **{pred_imp}** ({conf:.0%})")
        else:
            st.error(f"Prediction: **{pred_imp}** ({conf:.0%}) - True: {modulation}")

# Active impairments summary
if has_impairments:
    st.markdown("---")
    active = []
    if cfo_hz > 0:
        active.append(f"CFO: {cfo_hz} Hz")
    if iq_amp_db > 0:
        active.append(f"I/Q Amp: {iq_amp_db:.1f} dB")
    if iq_phase_deg > 0:
        active.append(f"I/Q Phase: {iq_phase_deg:.0f}°")
    if dc_offset > 0:
        active.append(f"DC: {dc_offset:.2f}")
    if phase_noise_std > 0:
        active.append(f"Phase Noise: {phase_noise_std:.3f}")
    if fading_type == "rayleigh":
        active.append("Rayleigh fading")
    elif fading_type == "rician":
        active.append(f"Rician fading (K={k_factor:.0f})")

    st.info("**Active impairments:** " + " | ".join(active))

# Time-domain comparison (collapsible)
with st.expander("Time-Domain View", expanded=False):
    sample_idx = 0
    clean_sample = samples[sample_idx]
    impaired_sample = impaired_samples[sample_idx]
    t = np.arange(128)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    # I channel
    axes[0].plot(t, clean_sample[0], "b-", linewidth=1, alpha=0.8, label="Clean")
    if has_impairments:
        axes[0].plot(t, impaired_sample[0], "r-", linewidth=1, alpha=0.6, label="Impaired")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("I")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Q channel
    axes[1].plot(t, clean_sample[1], "b-", linewidth=1, alpha=0.8, label="Clean")
    if has_impairments:
        axes[1].plot(t, impaired_sample[1], "r-", linewidth=1, alpha=0.6, label="Impaired")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Q")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Information boxes (collapsible)
with st.expander("About Impairments", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **CFO (Carrier Frequency Offset)**
        Oscillator mismatch causes time-varying phase rotation. High CFO turns constellation into a ring.

        **I/Q Imbalance**
        Amplitude/phase mismatch between I and Q paths causes elliptical distortion.
        """)

    with col2:
        st.markdown("""
        **DC Offset**
        LO leakage shifts constellation away from origin.

        **Phase Noise**
        Oscillator instabilities spread constellation points angularly.
        """)