"""Domain Shift Demo - Observe accuracy collapse under various impairments."""

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
    load_model_by_name,
    get_available_datasets,
    get_available_models,
    predict_family,
    apply_impairments,
    apply_fading,
    normalize_signal,
)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

st.set_page_config(page_title="Domain Shift Demo", page_icon="📉", layout="wide")

st.title("Domain Shift Demo")
st.markdown(
    "Observe how classification accuracy degrades under increasingly severe "
    "impairments, demonstrating the **domain shift problem** in RF signal classification."
)

# Check for available models
available_models = get_available_models()
if not available_models:
    st.warning(
        "No trained models found. Train a model first:\n\n"
        "```bash\n"
        "uv run python scripts/train_pfcnn.py\n"
        "```"
    )
    st.stop()

# Sidebar configuration
st.sidebar.header("Configuration")

selected_model = st.sidebar.selectbox("Model", available_models)
model = load_model_by_name(selected_model)

if model is None:
    st.error(f"Failed to load model: {selected_model}")
    st.stop()

available_datasets = get_available_datasets()
selected_dataset = st.sidebar.selectbox("Dataset", available_datasets)
dataset = load_dataset(selected_dataset.lower())

if dataset is None:
    st.error(f"Failed to load dataset: {selected_dataset}")
    st.stop()

family_names = dataset["family_names"]

# Use test set
if "test" in dataset:
    test_data, test_labels, test_snrs = dataset["test"]
else:
    test_data, test_labels, test_snrs = dataset["val"]

if len(test_data) < 20:
    st.error("Not enough test samples (need at least 20).")
    st.stop()

st.sidebar.header("Evaluation Settings")
n_samples = st.sidebar.slider(
    "Samples per condition",
    min_value=20,
    max_value=min(200, len(test_data)),
    value=min(100, len(test_data)),
    step=20,
)

# Impairment sweep configurations
st.sidebar.header("Impairment Sweeps")

sweep_type = st.sidebar.selectbox(
    "Sweep Type",
    ["CFO (Carrier Frequency Offset)", "Phase Noise", "I/Q Imbalance", "SNR Degradation", "Combined Level"]
)


def get_sweep_params(sweep_type):
    """Get sweep parameters based on type."""
    if sweep_type == "CFO (Carrier Frequency Offset)":
        return {
            "name": "CFO",
            "param": "cfo_hz",
            "values": [0, 500, 1000, 2000, 3000, 4000, 5000],
            "unit": "Hz",
            "xlabel": "Carrier Frequency Offset (Hz)",
        }
    elif sweep_type == "Phase Noise":
        return {
            "name": "Phase Noise",
            "param": "phase_noise",
            "values": [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1],
            "unit": "rad/sample",
            "xlabel": "Phase Noise (rad/sample)",
        }
    elif sweep_type == "I/Q Imbalance":
        return {
            "name": "I/Q Imbalance",
            "param": "iq_amp",
            "values": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            "unit": "dB",
            "xlabel": "I/Q Amplitude Imbalance (dB)",
        }
    elif sweep_type == "SNR Degradation":
        return {
            "name": "SNR",
            "param": "snr_filter",
            "values": [20, 15, 10, 5, 0, -5, -10],
            "unit": "dB",
            "xlabel": "Maximum SNR (dB)",
        }
    else:  # Combined Level
        return {
            "name": "Impairment Level",
            "param": "level",
            "values": [0, 1, 2, 3, 4, 5],
            "unit": "",
            "xlabel": "Impairment Severity Level",
            "level_params": [
                {"cfo_hz": 0, "phase_noise": 0, "iq_amp": 0, "iq_phase": 0},
                {"cfo_hz": 500, "phase_noise": 0.01, "iq_amp": 0.5, "iq_phase": 2},
                {"cfo_hz": 1000, "phase_noise": 0.02, "iq_amp": 1.0, "iq_phase": 5},
                {"cfo_hz": 2000, "phase_noise": 0.03, "iq_amp": 1.5, "iq_phase": 8},
                {"cfo_hz": 3000, "phase_noise": 0.05, "iq_amp": 2.0, "iq_phase": 12},
                {"cfo_hz": 5000, "phase_noise": 0.08, "iq_amp": 3.0, "iq_phase": 15},
            ],
        }


sweep_params = get_sweep_params(sweep_type)


@st.cache_data
def run_sweep_evaluation(_model, model_name, _test_data, _test_labels, _test_snrs, _family_names, n_samples, _sweep_params, sweep_type):
    """Run evaluation across impairment sweep."""
    results = []

    for val_idx, val in enumerate(_sweep_params["values"]):
        # Sample data
        indices = np.random.choice(len(_test_data), size=n_samples, replace=False)

        correct = 0
        total = 0

        for idx in indices:
            signal = _test_data[idx]
            label = _test_labels[idx]
            snr = _test_snrs[idx]

            # Convert complex to I/Q if needed
            if np.iscomplexobj(signal):
                signal = np.stack([signal.real, signal.imag], axis=0).astype(np.float32)

            # Apply impairments based on sweep type
            if _sweep_params["param"] == "snr_filter":
                # Filter by SNR
                if snr > val:
                    continue
            elif _sweep_params["param"] == "level":
                # Combined level
                level_p = _sweep_params["level_params"][val_idx]
                signal = apply_impairments(
                    signal,
                    cfo_hz=level_p["cfo_hz"],
                    iq_amp_db=level_p["iq_amp"],
                    iq_phase_deg=level_p["iq_phase"],
                    phase_noise_std=level_p["phase_noise"],
                )
                signal = normalize_signal(signal)
            else:
                # Single parameter sweep
                kwargs = {
                    "cfo_hz": 0,
                    "iq_amp_db": 0,
                    "iq_phase_deg": 0,
                    "phase_noise_std": 0,
                }
                if _sweep_params["param"] == "cfo_hz":
                    kwargs["cfo_hz"] = val
                elif _sweep_params["param"] == "phase_noise":
                    kwargs["phase_noise_std"] = val
                elif _sweep_params["param"] == "iq_amp":
                    kwargs["iq_amp_db"] = val
                    kwargs["iq_phase_deg"] = val * 3  # Scale phase with amplitude

                signal = apply_impairments(signal, **kwargs)
                signal = normalize_signal(signal)

            # Run prediction
            pred_name, probs = predict_family(_model, signal, _family_names)
            if pred_name is not None:
                pred_idx = _family_names.index(pred_name)
                if pred_idx == label:
                    correct += 1
                total += 1

        acc = correct / total if total > 0 else 0
        results.append({"value": val, "accuracy": acc, "n_samples": total})

    return results


# Run sweep
if st.button("Run Domain Shift Analysis", type="primary"):
    with st.spinner("Running impairment sweep..."):
        results = run_sweep_evaluation(
            model,
            selected_model,
            test_data,
            test_labels,
            test_snrs,
            family_names,
            n_samples,
            sweep_params,
            sweep_type,
        )
    st.session_state["sweep_results"] = results
    st.session_state["sweep_params"] = sweep_params

# Display results
if "sweep_results" in st.session_state:
    results = st.session_state["sweep_results"]
    params = st.session_state["sweep_params"]

    st.markdown("---")
    st.subheader(f"Accuracy vs {params['name']}")

    # Extract data
    values = [r["value"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    # Calculate domain gap
    if len(accuracies) >= 2:
        clean_acc = accuracies[0]
        worst_acc = min(accuracies)
        domain_gap = clean_acc - worst_acc
    else:
        clean_acc = worst_acc = domain_gap = 0

    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Clean Accuracy", f"{clean_acc:.1%}")
    col2.metric("Worst Accuracy", f"{worst_acc:.1%}")
    col3.metric("Domain Gap", f"{domain_gap:.1%}", delta=f"-{domain_gap:.1%}", delta_color="inverse")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(values, accuracies, 'o-', color='#2563eb', linewidth=2, markersize=8)
    ax.fill_between(values, accuracies, alpha=0.2, color='#2563eb')

    # Add reference lines
    ax.axhline(y=clean_acc, color='green', linestyle='--', alpha=0.5, label=f'Clean: {clean_acc:.1%}')
    ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Random (5 classes)')

    # Highlight domain gap
    if len(values) > 1:
        ax.annotate(
            f'Domain Gap: {domain_gap:.1%}',
            xy=(values[-1], worst_acc),
            xytext=(values[-1], (clean_acc + worst_acc) / 2),
            fontsize=10,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
            color='red',
        )

    ax.set_xlabel(params["xlabel"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Classification Accuracy vs {params['name']}")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Interpretation
    st.markdown("---")
    st.subheader("Interpretation")

    if domain_gap > 0.3:
        st.error(
            f"**Severe domain shift detected!** The model loses {domain_gap:.0%} accuracy "
            f"under these impairments. This indicates the model has not learned robust features "
            f"and is vulnerable to real-world conditions."
        )
    elif domain_gap > 0.15:
        st.warning(
            f"**Moderate domain shift.** The model shows {domain_gap:.0%} accuracy degradation. "
            f"Consider training with augmentation (MDA-DMC) to improve robustness."
        )
    else:
        st.success(
            f"**Good robustness!** The model maintains accuracy with only {domain_gap:.0%} "
            f"degradation. The features learned generalize well to impaired conditions."
        )

    # Show detailed results table
    with st.expander("Detailed Results"):
        import pandas as pd
        df = pd.DataFrame(results)
        df.columns = [params["xlabel"].split()[0], "Accuracy", "Samples"]
        df["Accuracy"] = df["Accuracy"].apply(lambda x: f"{x:.1%}")
        st.dataframe(df, use_container_width=True)

else:
    st.info("Click 'Run Domain Shift Analysis' to see how accuracy degrades under impairments.")

# Information section
st.markdown("---")
st.subheader("About Domain Shift")
st.markdown("""
**Domain shift** occurs when the test data distribution differs from training data.
In RF signal classification, this manifests as:

- **Hardware impairments**: CFO, I/Q imbalance, phase noise from real receivers
- **Channel effects**: Multipath fading, Doppler shift, delay spread
- **SNR variation**: Real-world SNR often differs from training conditions

The **domain gap** metric quantifies how much accuracy drops from clean to impaired conditions.
A robust classifier should have a small domain gap, indicating learned features generalize well.

**MDA-DMC** (Multi-Domain Augmentation for Domain-Mismatch Compensation) addresses this by
training with augmented data that simulates various impairments.
""")