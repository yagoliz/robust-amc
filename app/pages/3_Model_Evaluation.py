"""Model Evaluation - Test classifier performance."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from robust_amc.data import Compose, PowerNormalize, RadioMLDataset
from robust_amc.data.radioml2018_loader import SNR_LEVELS_2018
from robust_amc.data.radioml_loader import SNR_LEVELS
from robust_amc.data.transforms import ToTensor
from robust_amc.evaluation.metrics import (
    accuracy_by_snr,
    compute_confusion_matrix,
    evaluate_model,
)
from robust_amc.utils import get_device

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    DATA_PATH_2018,
    get_available_models,
    get_model_class_names,
    is_model_2018,
    load_dataset,
    load_model_by_name,
)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})

st.set_page_config(page_title="Model Evaluation", page_icon="🎯", layout="wide")

st.title("Model Evaluation")
st.markdown("Evaluate classifiers on their test sets.")

# Model selection
st.sidebar.header("Model Selection")
available_models = get_available_models()

if len(available_models) == 0:
    st.error("No trained models found. Please train at least one model first.")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", available_models)
model = load_model_by_name(selected_model)

if model is None:
    st.error(f"Failed to load model: {selected_model}")
    st.stop()

# Determine which dataset to use based on model
use_2018 = is_model_2018(selected_model)
class_names = get_model_class_names(selected_model)

if use_2018:
    dataset = load_dataset(DATA_PATH_2018, "2018")
    snr_levels = SNR_LEVELS_2018
    st.info(f"Using RadioML2018 test data (24 classes) for {selected_model}")
else:
    dataset = load_dataset()
    snr_levels = SNR_LEVELS

if dataset is None:
    if use_2018:
        st.error("RadioML2018 dataset not found. Please download or preprocess it first.")
    else:
        st.error("Dataset not found. Please download RadioML2016.10a.")
    st.stop()

# Display model info
st.sidebar.markdown("---")
st.sidebar.header("Model Info")
n_params = sum(p.numel() for p in model.parameters())
st.sidebar.metric("Parameters", f"{n_params:,}")
st.sidebar.metric("Classes", len(class_names))
st.sidebar.metric("Device", get_device())

# Get test data
test_data, test_labels, test_snrs = dataset["test"]
st.sidebar.metric("Test samples", f"{len(test_labels):,}")

# Run evaluation button
if st.button("Run Evaluation", type="primary"):
    with st.spinner("Evaluating..."):
        transform = Compose([PowerNormalize(), ToTensor()])
        test_dataset = RadioMLDataset(
            test_data, test_labels, test_snrs, transform=transform
        )
        test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=0
        )

        device = get_device()
        results = evaluate_model(model, test_loader, device)

        st.session_state["eval_results"] = results
        st.session_state["eval_complete"] = True

# Display results
if st.session_state.get("eval_complete", False):
    results = st.session_state["eval_results"]

    # Overall metrics
    st.subheader("Overall Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test Accuracy", f"{results['accuracy']:.1%}")

    with col2:
        high_snr_mask = results["snrs"] >= 10
        high_snr_acc = (results["predictions"][high_snr_mask] == results["targets"][high_snr_mask]).mean()
        st.metric("High SNR (≥10 dB)", f"{high_snr_acc:.1%}")

    with col3:
        low_snr_mask = results["snrs"] <= 0
        low_snr_acc = (results["predictions"][low_snr_mask] == results["targets"][low_snr_mask]).mean()
        st.metric("Low SNR (≤0 dB)", f"{low_snr_acc:.1%}")

    # Accuracy vs SNR
    st.subheader("Accuracy vs SNR")

    snr_acc = accuracy_by_snr(results["targets"], results["predictions"], results["snrs"])
    snr_values = sorted(snr_acc.keys())
    accuracies = [snr_acc[snr] for snr in snr_values]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(snr_values, accuracies, "o-", linewidth=2, markersize=6, color="#2563eb")
    ax.axhline(y=results["accuracy"], color="#dc2626", linestyle="--",
               linewidth=1.5, label=f"Overall: {results['accuracy']:.1%}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.05])
    ax.set_xticks(snr_values)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    snr_filter = st.selectbox(
        "SNR Range",
        ["All", "High (≥10 dB)", "Low (≤0 dB)"],
    )

    if snr_filter == "All":
        mask = np.ones(len(results["targets"]), dtype=bool)
    elif snr_filter == "High (≥10 dB)":
        mask = results["snrs"] >= 10
    else:
        mask = results["snrs"] <= 0

    cm = compute_confusion_matrix(
        results["targets"][mask],
        results["predictions"][mask],
        normalize=True,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a cleaner colormap with white for low values
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    # Simplified labels - only show values > 0.05 to reduce clutter
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            if val > 0.05:  # Only show significant values
                color = "white" if val > 0.5 else "black"
                # Show as percentage without decimal for cleaner look
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                       color=color, fontsize=9, fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Per-class accuracy
    st.subheader("Per-Class Accuracy")

    class_acc = {}
    for i, mod in enumerate(class_names):
        class_mask = results["targets"] == i
        if class_mask.sum() > 0:
            class_acc[mod] = (results["predictions"][class_mask] == i).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    mods = list(class_acc.keys())
    accs = list(class_acc.values())

    # Color by performance
    colors = ["#22c55e" if a >= 0.8 else "#f59e0b" if a >= 0.6 else "#ef4444" for a in accs]

    bars = ax.bar(mods, accs, color=colors, alpha=0.8, edgecolor="none")
    ax.axhline(y=results["accuracy"], color="#2563eb", linestyle="--",
               linewidth=1.5, label=f"Overall: {results['accuracy']:.1%}")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.1])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2, axis="y")

    # Add value labels only for bars below threshold
    for bar, acc in zip(bars, accs):
        if acc < 0.9:  # Only label bars that might need attention
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f"{acc:.0%}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Analysis
    with st.expander("Analysis", expanded=True):
        col1, col2 = st.columns(2)

        sorted_acc = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)

        with col1:
            st.markdown("**Best performing:**")
            for mod, acc in sorted_acc[:3]:
                st.markdown(f"- {mod}: {acc:.0%}")

        with col2:
            st.markdown("**Most challenging:**")
            for mod, acc in sorted_acc[-3:]:
                st.markdown(f"- {mod}: {acc:.0%}")

        # Common confusions
        st.markdown("**Common confusions** (>10%):")
        confusions = []
        for i, mod_true in enumerate(class_names):
            for j, mod_pred in enumerate(class_names):
                if i != j and cm[i, j] > 0.1:
                    confusions.append((mod_true, mod_pred, cm[i, j]))

        if confusions:
            confusions.sort(key=lambda x: x[2], reverse=True)
            for true, pred, rate in confusions[:5]:
                st.markdown(f"- {true} → {pred}: {rate:.0%}")
        else:
            st.markdown("- None (all confusions below 10%)")

else:
    st.info("Click **Run Evaluation** to test the model.")