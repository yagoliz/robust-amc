"""Model Comparison - Compare baseline vs augmented models."""

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
    MODEL_CHECKPOINTS,
)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.title("Model Comparison")
st.markdown(
    "Compare classification performance between baseline and augmented models "
    "under clean and impaired conditions."
)

# Check for available models
available_models = get_available_models()
if len(available_models) < 1:
    st.warning(
        "No trained models found. Train models first:\n\n"
        "```bash\n"
        "# Baseline\n"
        "uv run python scripts/train_pfcnn.py\n\n"
        "# With augmentation\n"
        "uv run python scripts/train_pfcnn.py --augment\n"
        "```"
    )
    st.stop()

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
selected_models = st.sidebar.multiselect(
    "Models to compare",
    available_models,
    default=available_models[:min(2, len(available_models))],
)

if len(selected_models) < 1:
    st.info("Select at least one model to compare.")
    st.stop()

# Load models
models = {}
for model_name in selected_models:
    model = load_model_by_name(model_name)
    if model is not None:
        models[model_name] = model

if not models:
    st.error("Failed to load any models.")
    st.stop()

# Dataset selection
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

# Evaluation settings
if len(test_data) < 50:
    st.error("Not enough test samples (need at least 50).")
    st.stop()

st.sidebar.header("Evaluation Settings")
n_samples = st.sidebar.slider(
    "Samples per condition",
    min_value=50,
    max_value=min(500, len(test_data)),
    value=min(200, len(test_data)),
    step=50,
)

# Impairment levels for comparison
st.sidebar.header("Test Conditions")
test_clean = st.sidebar.checkbox("Clean (no impairments)", value=True)
test_mild = st.sidebar.checkbox("Mild impairments", value=True)
test_severe = st.sidebar.checkbox("Severe impairments", value=True)

# Define impairment levels
IMPAIRMENT_LEVELS = {
    "Clean": {"cfo_hz": 0, "iq_amp_db": 0, "iq_phase_deg": 0, "phase_noise_std": 0},
    "Mild": {"cfo_hz": 1000, "iq_amp_db": 1.0, "iq_phase_deg": 5, "phase_noise_std": 0.02},
    "Severe": {"cfo_hz": 3000, "iq_amp_db": 2.5, "iq_phase_deg": 12, "phase_noise_std": 0.05},
}


@st.cache_data
def evaluate_model_condition(_model, model_name, _test_data, _test_labels, _family_names, n_samples, condition_name, impairments):
    """Evaluate a model under specific impairment conditions."""
    indices = np.random.choice(len(_test_data), size=n_samples, replace=False)

    predictions = []
    ground_truth = []

    for idx in indices:
        signal = _test_data[idx]
        label = _test_labels[idx]

        # Convert complex to I/Q if needed
        if np.iscomplexobj(signal):
            signal = np.stack([signal.real, signal.imag], axis=0).astype(np.float32)

        # Apply impairments
        if any(v != 0 for v in impairments.values()):
            signal = apply_impairments(signal, **impairments)
            signal = normalize_signal(signal)

        # Run prediction
        pred_name, probs = predict_family(_model, signal, _family_names)
        if pred_name is not None:
            pred_idx = _family_names.index(pred_name)
            predictions.append(pred_idx)
            ground_truth.append(label)

    accuracy = accuracy_score(ground_truth, predictions) if predictions else 0
    return accuracy, np.array(predictions), np.array(ground_truth)


# Run comparison
if st.button("Run Comparison", type="primary"):
    # Determine which conditions to test
    conditions = []
    if test_clean:
        conditions.append("Clean")
    if test_mild:
        conditions.append("Mild")
    if test_severe:
        conditions.append("Severe")

    if not conditions:
        st.warning("Select at least one test condition.")
        st.stop()

    results = {}
    progress = st.progress(0)
    total_evals = len(models) * len(conditions)
    current = 0

    for model_name, model in models.items():
        results[model_name] = {}
        for condition in conditions:
            with st.spinner(f"Evaluating {model_name} on {condition}..."):
                impairments = IMPAIRMENT_LEVELS[condition]
                acc, preds, gt = evaluate_model_condition(
                    model, model_name, test_data, test_labels, family_names, n_samples, condition, impairments
                )
                results[model_name][condition] = {
                    "accuracy": acc,
                    "predictions": preds,
                    "ground_truth": gt,
                }
            current += 1
            progress.progress(current / total_evals)

    st.session_state["comparison_results"] = results
    st.session_state["comparison_conditions"] = conditions
    progress.empty()

# Display results
if "comparison_results" in st.session_state:
    results = st.session_state["comparison_results"]
    conditions = st.session_state["comparison_conditions"]

    st.markdown("---")
    st.subheader("Accuracy Comparison")

    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(conditions))
    width = 0.8 / len(results)
    colors = ['#2563eb', '#dc2626', '#22c55e', '#f59e0b']

    for i, (model_name, model_results) in enumerate(results.items()):
        accuracies = [model_results[c]["accuracy"] for c in conditions]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=model_name, color=colors[i % len(colors)], alpha=0.8)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'{acc:.1%}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Test Condition')
    ax.set_title('Model Accuracy by Test Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Metrics table
    st.markdown("---")
    st.subheader("Detailed Metrics")

    import pandas as pd

    # Build metrics dataframe
    metrics_data = []
    for model_name in results.keys():
        row = {"Model": model_name}
        for condition in conditions:
            acc = results[model_name][condition]["accuracy"]
            row[f"{condition} Acc"] = f"{acc:.1%}"
        metrics_data.append(row)

    # Calculate robustness (if clean and severe exist)
    if "Clean" in conditions and "Severe" in conditions:
        for row in metrics_data:
            model_name = row["Model"]
            clean_acc = results[model_name]["Clean"]["accuracy"]
            severe_acc = results[model_name]["Severe"]["accuracy"]
            gap = clean_acc - severe_acc
            row["Robustness Gap"] = f"{gap:.1%}"

    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Robustness analysis
    if "Clean" in conditions and "Severe" in conditions and len(results) > 1:
        st.markdown("---")
        st.subheader("Robustness Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Domain Gap Comparison**")

            gaps = {}
            for model_name in results.keys():
                clean_acc = results[model_name]["Clean"]["accuracy"]
                severe_acc = results[model_name]["Severe"]["accuracy"]
                gaps[model_name] = clean_acc - severe_acc

            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.barh(list(gaps.keys()), list(gaps.values()), color='#ef4444', alpha=0.8)
            ax.set_xlabel("Domain Gap (Clean - Severe)")
            ax.set_title("Accuracy Drop Under Severe Impairments")
            ax.set_xlim(0, max(gaps.values()) * 1.2 if gaps.values() else 0.5)

            for bar, gap in zip(bars, gaps.values()):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                       f'{gap:.1%}', va='center', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("**Interpretation**")

            # Find best model
            best_model = min(gaps.keys(), key=lambda m: gaps[m])
            worst_model = max(gaps.keys(), key=lambda m: gaps[m])

            if gaps[best_model] < gaps[worst_model] - 0.05:
                st.success(
                    f"**{best_model}** shows better robustness with only "
                    f"{gaps[best_model]:.1%} accuracy drop under severe impairments, "
                    f"compared to {gaps[worst_model]:.1%} for {worst_model}."
                )
            else:
                st.info(
                    "Both models show similar robustness characteristics. "
                    "The domain gap is comparable across models."
                )

            # Show relative improvement
            if len(gaps) >= 2:
                sorted_models = sorted(gaps.keys(), key=lambda m: gaps[m])
                improvement = gaps[sorted_models[-1]] - gaps[sorted_models[0]]
                if improvement > 0.05:
                    st.markdown(
                        f"Training with augmentation reduces the domain gap by "
                        f"**{improvement:.1%}** compared to the baseline."
                    )

    # Confusion matrices (side by side for 2 models)
    if len(results) <= 3 and "Clean" in conditions:
        st.markdown("---")
        st.subheader("Confusion Matrices (Clean Condition)")

        cols = st.columns(len(results))
        for col, (model_name, model_results) in zip(cols, results.items()):
            with col:
                st.markdown(f"**{model_name}**")

                preds = model_results["Clean"]["predictions"]
                gt = model_results["Clean"]["ground_truth"]

                cm = confusion_matrix(gt, preds, labels=range(len(family_names)))
                cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

                fig, ax = plt.subplots(figsize=(4, 3.5))
                im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

                ax.set_xticks(range(len(family_names)))
                ax.set_yticks(range(len(family_names)))
                ax.set_xticklabels(family_names, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(family_names, fontsize=8)

                for i in range(len(family_names)):
                    for j in range(len(family_names)):
                        value = cm_norm[i, j]
                        color = "white" if value > 0.5 else "black"
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                               color=color, fontsize=7)

                ax.set_xlabel("Predicted", fontsize=9)
                ax.set_ylabel("True", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

else:
    st.info("Select models and click 'Run Comparison' to compare performance.")

# Information section
st.markdown("---")
st.subheader("About Model Comparison")
st.markdown("""
This page compares the performance of different trained models under various conditions:

**Test Conditions:**
- **Clean**: No additional impairments applied
- **Mild**: Moderate CFO (1000 Hz), I/Q imbalance (1 dB), phase noise (0.02 rad/sample)
- **Severe**: Strong CFO (3000 Hz), I/Q imbalance (2.5 dB), phase noise (0.05 rad/sample)

**Key Metrics:**
- **Accuracy**: Classification accuracy under each condition
- **Robustness Gap**: Accuracy drop from clean to severe conditions

A robust model should:
1. Have high accuracy on clean data
2. Maintain accuracy under impairments (small robustness gap)

Models trained with **MDA-DMC augmentation** typically show smaller robustness gaps
because they've learned features that are invariant to the impairments simulated
during training.
""")