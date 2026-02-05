"""Impairment Simulator - See how hardware impairments affect signals."""

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
    apply_impairments,
    apply_fading,
    normalize_signal,
    load_model,
    predict_family,
)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

AXIS_LIMIT = 3.0

st.set_page_config(page_title="Impairment Simulator", page_icon="🔧", layout="wide")

st.title("Impairment Simulator")
st.markdown("Observe how hardware impairments affect signal quality in real-time.")

# Load dataset
available_datasets = get_available_datasets()
if not available_datasets:
    st.error("No datasets available.")
    st.stop()

selected_dataset = st.sidebar.selectbox("Dataset", available_datasets)
dataset = load_dataset(selected_dataset.lower())

if dataset is None:
    st.error(f"Failed to load {selected_dataset}")
    st.stop()

# Load model (optional)
model = load_model()

family_names = dataset["family_names"]

# Use test set if available
if "test" in dataset:
    test_data, test_labels, test_snrs = dataset["test"]
elif "val" in dataset:
    test_data, test_labels, test_snrs = dataset["val"]
else:
    st.error("No test data available")
    st.stop()

# Sidebar - Signal Selection
st.sidebar.header("Signal Selection")

family = st.sidebar.selectbox("Modulation Family", family_names)
family_idx = family_names.index(family)

snr_list = sorted(set(int(s) for s in test_snrs))
snr = st.sidebar.select_slider("SNR (dB)", options=snr_list, value=10 if 10 in snr_list else snr_list[len(snr_list)//2])

# Sidebar - Impairments
st.sidebar.header("Impairments")

cfo_hz = st.sidebar.slider("Carrier Frequency Offset (Hz)", -5000, 5000, 0, 100)
iq_amp = st.sidebar.slider("I/Q Amplitude Imbalance (dB)", 0.0, 5.0, 0.0, 0.1)
iq_phase = st.sidebar.slider("I/Q Phase Imbalance (deg)", 0.0, 20.0, 0.0, 0.5)
dc_offset = st.sidebar.slider("DC Offset (relative)", 0.0, 0.3, 0.0, 0.01)
phase_noise = st.sidebar.slider("Phase Noise (rad/sample)", 0.0, 0.1, 0.0, 0.005)

# Sidebar - Fading
st.sidebar.header("Fading")
fading_type = st.sidebar.selectbox("Fading Type", ["none", "rayleigh", "rician"])
k_factor = 1.0
if fading_type == "rician":
    k_factor = st.sidebar.slider("Rician K-factor", 0.0, 20.0, 5.0, 0.5)

# Get sample
samples = get_samples_for_family(test_data, test_labels, test_snrs, family_idx, snr, 1)

if samples is None:
    st.warning(f"No samples found for {family} at {snr} dB")
    st.stop()

signal = samples[0]

# Apply impairments
impaired_signal = apply_impairments(
    signal,
    cfo_hz=cfo_hz,
    iq_amp_db=iq_amp,
    iq_phase_deg=iq_phase,
    dc_i=dc_offset,
    dc_q=dc_offset,
    phase_noise_std=phase_noise,
)

# Apply fading
impaired_signal = apply_fading(impaired_signal, fading_type, k_factor)

# Normalize for display
impaired_signal = normalize_signal(impaired_signal)

# Display
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Signal")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(signal[0].flatten(), signal[1].flatten(), alpha=0.5, s=8, c="#2563eb")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(f"{family} @ {snr} dB (Clean)")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Model prediction on clean signal
    if model is not None:
        pred, probs = predict_family(model, signal, family_names)
        if pred:
            st.metric("Prediction (Clean)", pred, delta=None)

with col2:
    st.subheader("Impaired Signal")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(impaired_signal[0].flatten(), impaired_signal[1].flatten(), alpha=0.5, s=8, c="#dc2626")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(f"{family} @ {snr} dB (Impaired)")
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Model prediction on impaired signal
    if model is not None:
        pred_imp, probs_imp = predict_family(model, impaired_signal, family_names)
        if pred_imp:
            correct = pred_imp == family
            st.metric("Prediction (Impaired)", pred_imp,
                     delta="Correct" if correct else "Wrong",
                     delta_color="normal" if correct else "inverse")

# Time domain comparison
st.markdown("---")
st.subheader("Time Domain Comparison")

fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

seq_len = signal.shape[1]
t = np.arange(seq_len)

# Original
axes[0, 0].plot(t, signal[0], "-", linewidth=1, color="#2563eb")
axes[0, 0].set_ylabel("I (Original)")
axes[0, 0].grid(True, alpha=0.2)

axes[1, 0].plot(t, signal[1], "-", linewidth=1, color="#2563eb")
axes[1, 0].set_ylabel("Q (Original)")
axes[1, 0].set_xlabel("Sample")
axes[1, 0].grid(True, alpha=0.2)

# Impaired
axes[0, 1].plot(t, impaired_signal[0], "-", linewidth=1, color="#dc2626")
axes[0, 1].set_ylabel("I (Impaired)")
axes[0, 1].grid(True, alpha=0.2)

axes[1, 1].plot(t, impaired_signal[1], "-", linewidth=1, color="#dc2626")
axes[1, 1].set_ylabel("Q (Impaired)")
axes[1, 1].set_xlabel("Sample")
axes[1, 1].grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Active impairments summary
st.sidebar.markdown("---")
st.sidebar.markdown("### Active Impairments")
active = []
if cfo_hz != 0:
    active.append(f"CFO: {cfo_hz} Hz")
if iq_amp > 0:
    active.append(f"I/Q Amp: {iq_amp} dB")
if iq_phase > 0:
    active.append(f"I/Q Phase: {iq_phase}°")
if dc_offset > 0:
    active.append(f"DC: {dc_offset:.2f}")
if phase_noise > 0:
    active.append(f"Phase Noise: {phase_noise:.3f}")
if fading_type != "none":
    active.append(f"Fading: {fading_type}")

if active:
    for imp in active:
        st.sidebar.write(f"- {imp}")
else:
    st.sidebar.write("None (clean signal)")
