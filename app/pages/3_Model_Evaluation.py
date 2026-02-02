"""Model Evaluation - Test classifier performance."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from robust_amc.data import RadioMLDataset, PowerNormalize, Compose
from robust_amc.data.transforms import ToTensor
from robust_amc.data.radioml_loader import MODULATION_CLASSES, SNR_LEVELS
from robust_amc.evaluation.metrics import (
    evaluate_model,
    accuracy_by_snr,
    compute_confusion_matrix,
)

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_dataset, load_model, get_device

st.set_page_config(page_title="Model Evaluation", page_icon="🎯", layout="wide")

st.title("Model Evaluation")
st.markdown("Evaluate the PF-CNN classifier on the test set.")

# Load data and model
dataset = load_dataset()
model = load_model()

if dataset is None:
    st.error("Dataset not found. Please download RadioML2016.10a.")
    st.stop()

if model is None:
    st.error("Model not found. Please train the baseline model first.")
    st.stop()

# Display model info
st.sidebar.header("Model Information")
n_params = sum(p.numel() for p in model.parameters())
st.sidebar.metric("Parameters", f"{n_params:,}")
st.sidebar.metric("Device", get_device())

# Evaluation options
st.sidebar.header("Evaluation Options")
batch_size = st.sidebar.slider("Batch Size", 64, 512, 256, 64)

# Get test data
test_data, test_labels, test_snrs = dataset["test"]
st.sidebar.metric("Test Samples", f"{len(test_labels):,}")

# Run evaluation button
if st.button("Run Evaluation", type="primary"):
    with st.spinner("Evaluating model..."):
        # Create dataset and loader
        transform = Compose([PowerNormalize(), ToTensor()])
        test_dataset = RadioMLDataset(
            test_data, test_labels, test_snrs, transform=transform
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Run evaluation
        device = get_device()
        results = evaluate_model(model, test_loader, device)

        # Store in session state
        st.session_state["eval_results"] = results
        st.session_state["eval_complete"] = True

# Display results if available
if st.session_state.get("eval_complete", False):
    results = st.session_state["eval_results"]

    # Overall accuracy
    st.subheader("Overall Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test Accuracy", f"{results['accuracy']:.2%}")

    with col2:
        # Accuracy at high SNR (>= 10 dB)
        high_snr_mask = results["snrs"] >= 10
        high_snr_acc = (
            results["predictions"][high_snr_mask] == results["targets"][high_snr_mask]
        ).mean()
        st.metric("Accuracy (SNR >= 10 dB)", f"{high_snr_acc:.2%}")

    with col3:
        # Accuracy at low SNR (<= 0 dB)
        low_snr_mask = results["snrs"] <= 0
        low_snr_acc = (
            results["predictions"][low_snr_mask] == results["targets"][low_snr_mask]
        ).mean()
        st.metric("Accuracy (SNR <= 0 dB)", f"{low_snr_acc:.2%}")

    # Accuracy vs SNR
    st.subheader("Accuracy vs SNR")

    snr_acc = accuracy_by_snr(results["targets"], results["predictions"], results["snrs"])
    snr_values = sorted(snr_acc.keys())
    accuracies = [snr_acc[snr] for snr in snr_values]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(snr_values, accuracies, "b-o", linewidth=2, markersize=8)
    ax.axhline(y=results["accuracy"], color="r", linestyle="--", label=f"Overall: {results['accuracy']:.2%}")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Classification Accuracy vs SNR", fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.set_xticks(snr_values)
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # SNR table
    with st.expander("SNR Accuracy Table"):
        snr_data = {"SNR (dB)": snr_values, "Accuracy": [f"{acc:.2%}" for acc in accuracies]}
        st.table(snr_data)

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    snr_filter = st.selectbox(
        "Filter by SNR",
        ["All SNRs", "High SNR (>= 10 dB)", "Mid SNR (0-10 dB)", "Low SNR (<= 0 dB)"],
    )

    if snr_filter == "All SNRs":
        mask = np.ones(len(results["targets"]), dtype=bool)
    elif snr_filter == "High SNR (>= 10 dB)":
        mask = results["snrs"] >= 10
    elif snr_filter == "Mid SNR (0-10 dB)":
        mask = (results["snrs"] > 0) & (results["snrs"] < 10)
    else:
        mask = results["snrs"] <= 0

    cm = compute_confusion_matrix(
        results["targets"][mask],
        results["predictions"][mask],
        normalize=True,
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(MODULATION_CLASSES)))
    ax.set_yticks(range(len(MODULATION_CLASSES)))
    ax.set_xticklabels(MODULATION_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(MODULATION_CLASSES)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix ({snr_filter})", fontsize=14)

    # Add text annotations
    for i in range(len(MODULATION_CLASSES)):
        for j in range(len(MODULATION_CLASSES)):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Per-class accuracy
    st.subheader("Per-Class Performance")

    class_acc = {}
    for i, mod in enumerate(MODULATION_CLASSES):
        class_mask = results["targets"] == i
        if class_mask.sum() > 0:
            class_acc[mod] = (results["predictions"][class_mask] == i).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    mods = list(class_acc.keys())
    accs = list(class_acc.values())
    colors = ["green" if a >= 0.8 else "orange" if a >= 0.6 else "red" for a in accs]

    bars = ax.bar(mods, accs, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(y=results["accuracy"], color="blue", linestyle="--", label=f"Overall: {results['accuracy']:.2%}")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend()

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Hardest modulations
    st.subheader("Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Easiest Modulations** (highest accuracy)")
        sorted_acc = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)
        for mod, acc in sorted_acc[:3]:
            st.markdown(f"- {mod}: {acc:.1%}")

    with col2:
        st.markdown("**Hardest Modulations** (lowest accuracy)")
        for mod, acc in sorted_acc[-3:]:
            st.markdown(f"- {mod}: {acc:.1%}")

    # Common confusions
    st.markdown("**Common Confusions**")
    confusions = []
    for i, mod_true in enumerate(MODULATION_CLASSES):
        for j, mod_pred in enumerate(MODULATION_CLASSES):
            if i != j and cm[i, j] > 0.1:
                confusions.append((mod_true, mod_pred, cm[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)

    if confusions:
        for true, pred, rate in confusions[:5]:
            st.markdown(f"- {true} → {pred}: {rate:.1%}")
    else:
        st.markdown("No significant confusions (all < 10%)")

else:
    st.info("Click 'Run Evaluation' to evaluate the model on the test set.")

    # Show placeholder charts
    st.subheader("Expected Output")
    st.markdown("""
    After evaluation, you will see:
    - Overall test accuracy
    - Accuracy vs SNR curve
    - Confusion matrix (filterable by SNR range)
    - Per-class performance analysis
    """)