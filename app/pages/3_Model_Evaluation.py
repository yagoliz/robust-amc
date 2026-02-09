"""Model Evaluation - Test classifier on clean and impaired signals."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix

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
    normalize_signal,
)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

st.set_page_config(page_title="Model Evaluation", page_icon="🎯", layout="wide")

st.title("Model Evaluation")
st.markdown("Test the classifier on clean and impaired signals.")

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

# Sidebar - Model and Dataset Selection
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

# Use test set if available
if "test" in dataset:
    test_data, test_labels, test_snrs = dataset["test"]
else:
    test_data, test_labels, test_snrs = dataset["val"]

# Sidebar - Evaluation Settings
if len(test_data) < 50:
    st.error("Not enough test samples (need at least 50).")
    st.stop()

st.sidebar.header("Evaluation Settings")

n_samples = st.sidebar.slider(
    "Samples to evaluate",
    min_value=50,
    max_value=min(1000, len(test_data)),
    value=min(200, len(test_data)),
    step=50,
)

# Sidebar - Optional Impairments
st.sidebar.header("Test-Time Impairments")
apply_test_impairments = st.sidebar.checkbox("Apply impairments", value=False)

cfo_hz = 0
iq_amp = 0
iq_phase = 0
phase_noise = 0

if apply_test_impairments:
    cfo_hz = st.sidebar.slider("CFO (Hz)", -2000, 2000, 0, 100)
    iq_amp = st.sidebar.slider("I/Q Amp Imbalance (dB)", 0.0, 3.0, 0.0, 0.1)
    iq_phase = st.sidebar.slider("I/Q Phase Imbalance (deg)", 0.0, 15.0, 0.0, 0.5)
    phase_noise = st.sidebar.slider("Phase Noise (rad/sample)", 0.0, 0.05, 0.0, 0.005)


@st.cache_data
def run_evaluation(
    _model,
    model_name,
    _test_data,
    _test_labels,
    _test_snrs,
    _family_names,
    n_samples,
    cfo_hz,
    iq_amp,
    iq_phase,
    phase_noise,
    apply_impairments_flag,
):
    """Run model evaluation on test data."""
    indices = np.random.choice(len(_test_data), size=n_samples, replace=False)

    predictions = []
    ground_truth = []
    snrs = []
    confidences = []

    for idx in indices:
        signal = _test_data[idx]
        label = _test_labels[idx]
        snr = _test_snrs[idx]

        # Convert complex to I/Q if needed
        if np.iscomplexobj(signal):
            signal = np.stack([signal.real, signal.imag], axis=0).astype(np.float32)

        # Apply test-time impairments if enabled
        if apply_impairments_flag and (cfo_hz != 0 or iq_amp > 0 or iq_phase > 0 or phase_noise > 0):
            signal = apply_impairments(
                signal,
                cfo_hz=cfo_hz,
                iq_amp_db=iq_amp,
                iq_phase_deg=iq_phase,
                phase_noise_std=phase_noise,
            )
            signal = normalize_signal(signal)

        # Run prediction
        pred_name, probs = predict_family(_model, signal, _family_names)
        if pred_name is not None:
            pred_idx = _family_names.index(pred_name)
            predictions.append(pred_idx)
            ground_truth.append(label)
            snrs.append(snr)
            confidences.append(np.max(probs))

    return np.array(predictions), np.array(ground_truth), np.array(snrs), np.array(confidences)


# Run evaluation
if st.button("Run Evaluation", type="primary"):
    with st.spinner("Evaluating model..."):
        predictions, ground_truth, snrs, confidences = run_evaluation(
            model,
            selected_model,
            test_data,
            test_labels,
            test_snrs,
            family_names,
            n_samples,
            cfo_hz,
            iq_amp,
            iq_phase,
            phase_noise,
            apply_test_impairments,
        )

    # Store results in session state
    st.session_state["eval_results"] = {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "snrs": snrs,
        "confidences": confidences,
    }

# Display results if available
if "eval_results" in st.session_state:
    results = st.session_state["eval_results"]
    predictions = results["predictions"]
    ground_truth = results["ground_truth"]
    snrs = results["snrs"]
    confidences = results["confidences"]

    # Overall metrics
    st.markdown("---")
    st.subheader("Overall Performance")

    accuracy = accuracy_score(ground_truth, predictions)

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{accuracy:.1%}")
    col2.metric("Samples Evaluated", len(predictions))
    col3.metric("Mean Confidence", f"{np.mean(confidences):.1%}")

    # Confusion Matrix
    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(ground_truth, predictions, labels=range(len(family_names)))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)

        # Add labels
        ax.set_xticks(range(len(family_names)))
        ax.set_yticks(range(len(family_names)))
        ax.set_xticklabels(family_names, rotation=45, ha='right')
        ax.set_yticklabels(family_names)

        # Add text annotations
        for i in range(len(family_names)):
            for j in range(len(family_names)):
                value = cm_normalized[i, j]
                color = "white" if value > 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalized Confusion Matrix")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Per-Family Accuracy")

        per_family_acc = []
        for i, name in enumerate(family_names):
            mask = ground_truth == i
            if mask.sum() > 0:
                acc = (predictions[mask] == i).mean()
                per_family_acc.append((name, acc, mask.sum()))
            else:
                per_family_acc.append((name, 0.0, 0))

        # Display as bar chart
        fig, ax = plt.subplots(figsize=(5, 4))
        names = [x[0] for x in per_family_acc]
        accs = [x[1] for x in per_family_acc]
        counts = [x[2] for x in per_family_acc]

        bars = ax.barh(names, accs, color='#2563eb', alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title("Accuracy by Family")

        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f"n={count}", va='center', fontsize=9)

        ax.axvline(x=accuracy, color='red', linestyle='--', alpha=0.7, label=f'Overall: {accuracy:.1%}')
        ax.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Accuracy vs SNR
    st.markdown("---")
    st.subheader("Accuracy vs SNR")

    snr_bins = sorted(set(int(s) for s in snrs))

    if len(snr_bins) > 1:
        acc_by_snr = []
        for snr_val in snr_bins:
            mask = np.abs(snrs - snr_val) <= 1
            if mask.sum() > 0:
                acc = (predictions[mask] == ground_truth[mask]).mean()
                acc_by_snr.append((snr_val, acc, mask.sum()))

        fig, ax = plt.subplots(figsize=(10, 4))
        snr_vals = [x[0] for x in acc_by_snr]
        acc_vals = [x[1] for x in acc_by_snr]

        ax.plot(snr_vals, acc_vals, 'o-', color='#2563eb', linewidth=2, markersize=6)
        ax.fill_between(snr_vals, acc_vals, alpha=0.2, color='#2563eb')
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("Classification Accuracy vs SNR")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Random guess (5 classes)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough SNR variation to plot accuracy vs SNR.")

    # Confidence distribution
    st.markdown("---")
    st.subheader("Confidence Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confidence Distribution**")

        fig, ax = plt.subplots(figsize=(5, 4))

        correct_mask = predictions == ground_truth
        ax.hist(confidences[correct_mask], bins=20, alpha=0.7, label='Correct', color='#22c55e', range=(0, 1))
        ax.hist(confidences[~correct_mask], bins=20, alpha=0.7, label='Incorrect', color='#ef4444', range=(0, 1))

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Confidence")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("**Confidence Statistics**")

        correct_conf = confidences[correct_mask].mean() if correct_mask.sum() > 0 else 0
        incorrect_conf = confidences[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0

        st.write(f"- Mean confidence (correct): **{correct_conf:.1%}**")
        st.write(f"- Mean confidence (incorrect): **{incorrect_conf:.1%}**")
        st.write(f"- Confidence gap: **{correct_conf - incorrect_conf:.1%}**")

        # High confidence errors
        high_conf_errors = (confidences > 0.8) & ~correct_mask
        st.write(f"- High-confidence errors (>80%): **{high_conf_errors.sum()}** ({high_conf_errors.mean():.1%})")

else:
    st.info("Click 'Run Evaluation' to test the model.")