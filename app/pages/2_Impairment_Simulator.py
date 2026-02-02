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
)

st.set_page_config(page_title="Impairment Simulator", page_icon="⚡", layout="wide")

st.title("Impairment Simulator")
st.markdown("""
Explore how real-world hardware impairments affect I/Q signals and classifier performance.
Adjust the sliders to apply impairments and observe changes in real-time.
""")

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
st.sidebar.subheader("Carrier Frequency Offset")
cfo_enabled = st.sidebar.checkbox("Enable CFO", value=False)
cfo_hz = st.sidebar.slider(
    "CFO (Hz)",
    min_value=0,
    max_value=5000,
    value=0,
    step=100,
    disabled=not cfo_enabled,
)
if not cfo_enabled:
    cfo_hz = 0

# I/Q Imbalance
st.sidebar.subheader("I/Q Imbalance")
iq_enabled = st.sidebar.checkbox("Enable I/Q Imbalance", value=False)
iq_amp_db = st.sidebar.slider(
    "Amplitude Imbalance (dB)",
    min_value=0.0,
    max_value=3.0,
    value=0.0,
    step=0.1,
    disabled=not iq_enabled,
)
iq_phase_deg = st.sidebar.slider(
    "Phase Imbalance (deg)",
    min_value=0.0,
    max_value=15.0,
    value=0.0,
    step=0.5,
    disabled=not iq_enabled,
)
if not iq_enabled:
    iq_amp_db = 0.0
    iq_phase_deg = 0.0

# DC Offset
st.sidebar.subheader("DC Offset")
dc_enabled = st.sidebar.checkbox("Enable DC Offset", value=False)
dc_i = st.sidebar.slider(
    "DC Offset I (relative)",
    min_value=0.0,
    max_value=0.5,
    value=0.0,
    step=0.01,
    disabled=not dc_enabled,
)
dc_q = st.sidebar.slider(
    "DC Offset Q (relative)",
    min_value=0.0,
    max_value=0.5,
    value=0.0,
    step=0.01,
    disabled=not dc_enabled,
)
if not dc_enabled:
    dc_i = 0.0
    dc_q = 0.0

# Phase Noise
st.sidebar.subheader("Phase Noise")
pn_enabled = st.sidebar.checkbox("Enable Phase Noise", value=False)
phase_noise_std = st.sidebar.slider(
    "Phase Noise Std (rad/sample)",
    min_value=0.0,
    max_value=0.1,
    value=0.0,
    step=0.005,
    disabled=not pn_enabled,
)
if not pn_enabled:
    phase_noise_std = 0.0

# Fading
st.sidebar.markdown("---")
st.sidebar.header("Channel Fading")
fading_type = st.sidebar.selectbox(
    "Fading Type",
    ["none", "rayleigh", "rician"],
    index=0,
)
k_factor = st.sidebar.slider(
    "Rician K-factor",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=1.0,
    disabled=(fading_type != "rician"),
)

# Apply impairments to all samples
impaired_samples = []
for sample in samples:
    # Apply fading first
    faded = apply_fading(sample, fading_type, k_factor)
    # Then apply hardware impairments
    impaired = apply_impairments(
        faded,
        cfo_hz=cfo_hz,
        iq_amp_db=iq_amp_db,
        iq_phase_deg=iq_phase_deg,
        dc_i=dc_i,
        dc_q=dc_q,
        phase_noise_std=phase_noise_std,
    )
    impaired_samples.append(impaired)

impaired_samples = np.array(impaired_samples)

# Visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Signal")

    fig, ax = plt.subplots(figsize=(6, 6))
    I = samples[:, 0, :].flatten()
    Q = samples[:, 1, :].flatten()
    ax.scatter(I, Q, alpha=0.3, s=5, c="blue")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(f"{modulation} at {snr} dB (Clean)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    max_val = max(np.abs(I).max(), np.abs(Q).max()) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    st.pyplot(fig)
    plt.close(fig)

    # Model prediction on clean signal
    if model is not None:
        pred_clean, probs_clean = predict_modulation(model, samples[0])
        st.markdown(f"**Model Prediction (Clean):** {pred_clean}")
        correct_clean = pred_clean == modulation
        if correct_clean:
            st.success(f"Correct! Confidence: {probs_clean[MODULATION_CLASSES.index(modulation)]:.1%}")
        else:
            st.error(f"Incorrect! True: {modulation}, Confidence: {probs_clean[MODULATION_CLASSES.index(pred_clean)]:.1%}")

with col2:
    st.subheader("Impaired Signal")

    fig, ax = plt.subplots(figsize=(6, 6))
    I_imp = impaired_samples[:, 0, :].flatten()
    Q_imp = impaired_samples[:, 1, :].flatten()
    ax.scatter(I_imp, Q_imp, alpha=0.3, s=5, c="red")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")

    # Build title with active impairments
    impairment_list = []
    if cfo_hz > 0:
        impairment_list.append(f"CFO={cfo_hz}Hz")
    if iq_amp_db > 0 or iq_phase_deg > 0:
        impairment_list.append(f"IQ={iq_amp_db:.1f}dB/{iq_phase_deg:.1f}°")
    if dc_i > 0 or dc_q > 0:
        impairment_list.append(f"DC={dc_i:.2f}/{dc_q:.2f}")
    if phase_noise_std > 0:
        impairment_list.append(f"PN={phase_noise_std:.3f}")
    if fading_type != "none":
        if fading_type == "rician":
            impairment_list.append(f"Rician(K={k_factor:.0f})")
        else:
            impairment_list.append("Rayleigh")

    if impairment_list:
        title = f"{modulation} ({', '.join(impairment_list)})"
    else:
        title = f"{modulation} (No impairments)"

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Use same limits as clean signal for comparison
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    st.pyplot(fig)
    plt.close(fig)

    # Model prediction on impaired signal
    if model is not None:
        pred_imp, probs_imp = predict_modulation(model, impaired_samples[0])
        st.markdown(f"**Model Prediction (Impaired):** {pred_imp}")
        correct_imp = pred_imp == modulation
        if correct_imp:
            st.success(f"Correct! Confidence: {probs_imp[MODULATION_CLASSES.index(modulation)]:.1%}")
        else:
            st.error(f"Incorrect! True: {modulation}, Confidence: {probs_imp[MODULATION_CLASSES.index(pred_imp)]:.1%}")

# Time-domain comparison
st.subheader("Time-Domain Comparison")

sample_idx = 0
clean_sample = samples[sample_idx]
impaired_sample = impaired_samples[sample_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 6))
t = np.arange(128)

# Clean I
axes[0, 0].plot(t, clean_sample[0], "b-", linewidth=1, label="Clean")
axes[0, 0].plot(t, impaired_sample[0], "r-", linewidth=1, alpha=0.7, label="Impaired")
axes[0, 0].set_ylabel("In-phase (I)")
axes[0, 0].set_title("I Channel")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Clean Q
axes[0, 1].plot(t, clean_sample[1], "b-", linewidth=1, label="Clean")
axes[0, 1].plot(t, impaired_sample[1], "r-", linewidth=1, alpha=0.7, label="Impaired")
axes[0, 1].set_ylabel("Quadrature (Q)")
axes[0, 1].set_title("Q Channel")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Phase comparison
clean_phase = np.arctan2(clean_sample[1], clean_sample[0])
impaired_phase = np.arctan2(impaired_sample[1], impaired_sample[0])

axes[1, 0].plot(t, clean_phase, "b-", linewidth=1, label="Clean")
axes[1, 0].plot(t, impaired_phase, "r-", linewidth=1, alpha=0.7, label="Impaired")
axes[1, 0].set_xlabel("Sample")
axes[1, 0].set_ylabel("Phase (rad)")
axes[1, 0].set_title("Instantaneous Phase")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Amplitude comparison
clean_amp = np.sqrt(clean_sample[0] ** 2 + clean_sample[1] ** 2)
impaired_amp = np.sqrt(impaired_sample[0] ** 2 + impaired_sample[1] ** 2)

axes[1, 1].plot(t, clean_amp, "b-", linewidth=1, label="Clean")
axes[1, 1].plot(t, impaired_amp, "r-", linewidth=1, alpha=0.7, label="Impaired")
axes[1, 1].set_xlabel("Sample")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].set_title("Instantaneous Amplitude")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Information boxes
st.markdown("---")
st.subheader("Impairment Descriptions")

col1, col2 = st.columns(2)

with col1:
    with st.expander("Carrier Frequency Offset (CFO)"):
        st.markdown("""
        **Cause:** Mismatch between transmitter and receiver oscillator frequencies.

        **Effect:** Time-varying phase rotation that causes constellation to rotate.
        At high CFO values, the constellation becomes a ring or spiral.

        **Typical values:** 0-1000 Hz for well-calibrated systems, can be higher
        for low-cost receivers.
        """)

    with st.expander("I/Q Imbalance"):
        st.markdown("""
        **Cause:** Imperfect matching between I and Q signal paths in quadrature
        mixers (amplitude and phase differences).

        **Effect:**
        - Amplitude imbalance: Elliptical distortion of constellation
        - Phase imbalance: Skewing/rotation of constellation

        **Typical values:** 0.5-2 dB amplitude, 1-5° phase for practical receivers.
        """)

with col2:
    with st.expander("DC Offset"):
        st.markdown("""
        **Cause:** Local oscillator (LO) leakage, ADC offset, or component mismatch.

        **Effect:** Shifts the constellation away from the origin. Particularly
        problematic for low-amplitude modulations.

        **Typical values:** 0.1-5% of signal amplitude.
        """)

    with st.expander("Phase Noise"):
        st.markdown("""
        **Cause:** Oscillator instabilities causing random phase fluctuations.

        **Effect:** "Fuzzes" or spreads constellation points along the angular
        direction. More severe for higher-order modulations.

        **Typical values:** Varies with oscillator quality; low-cost oscillators
        have higher phase noise.
        """)