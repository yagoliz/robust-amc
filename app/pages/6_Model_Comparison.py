"""Model Comparison - Compare different model architectures and training methods."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from robust_amc.data.radioml_loader import MODULATION_CLASSES, SNR_LEVELS
from robust_amc.data.radioml2018_loader import (
    MODULATION_CLASSES_2018,
    CLASS_NAME_MAPPING_2018_TO_2016,
)
from robust_amc.data import PowerNormalize, Compose, get_data_loaders, OVERLAPPING_CLASSES
from robust_amc.data.transforms import ToTensor
from robust_amc.data.impairments import CarrierFrequencyOffset, IQImbalance, DCOffset
from robust_amc.data.channels import RayleighFading

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    load_dataset,
    get_available_models,
    get_available_datasets,
    load_model_by_name,
    get_samples_for_modulation,
    predict_modulation,
    normalize_samples,
    DATA_PATH,
    DATA_PATH_2018,
)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.title("Model Comparison")
st.markdown("""
Compare the performance of different models:
- **PF-CNN Baseline**: Standard supervised training
- **PF-CNN + MDA-DMC**: Trained with data augmentation
- **CLSR-AMC**: Contrastive learning with self-reconstruction
""")

# Load data
dataset = load_dataset()

if dataset is None:
    st.error("Dataset not found. Please download RadioML2016.10a.")
    st.stop()

# Check available models
available_models = get_available_models()

if len(available_models) == 0:
    st.error("No trained models found. Please train at least one model first.")
    st.info("""
    Train models with:
    ```bash
    # Baseline
    uv run python scripts/train_baseline.py

    # MDA-DMC
    uv run python scripts/train_mda_dmc.py

    # CLSR-AMC
    uv run python scripts/train_clsr_amc.py
    ```
    """)
    st.stop()

st.success(f"Found {len(available_models)} trained model(s): {', '.join(available_models)}")

# Model selection
st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect(
    "Select models to compare",
    available_models,
    default=available_models[:2] if len(available_models) > 1 else available_models,
)

if len(selected_models) == 0:
    st.warning("Please select at least one model to analyze.")
    st.stop()

# Load selected models
models = {}
for model_name in selected_models:
    model = load_model_by_name(model_name)
    if model is not None:
        models[model_name] = model

if len(models) == 0:
    st.error("Failed to load any models.")
    st.stop()

# Comparison type selection
st.sidebar.markdown("---")
st.sidebar.header("Comparison Type")
comparison_type = st.sidebar.selectbox(
    "What to compare",
    ["Single Signal Prediction", "SNR Sweep", "Impairment Robustness"],
)

# Get test data
test_data, test_labels, test_snrs = dataset["test"]

if comparison_type == "Single Signal Prediction":
    st.subheader("Single Signal Prediction Comparison")

    col1, col2 = st.columns(2)
    with col1:
        modulation = st.selectbox("Modulation Type", MODULATION_CLASSES, index=4)
    with col2:
        snr = st.select_slider("SNR (dB)", options=SNR_LEVELS, value=10)

    samples = get_samples_for_modulation(
        test_data, test_labels, test_snrs, modulation, snr, n_samples=5
    )

    if samples is None:
        st.warning(f"No samples found for {modulation} at {snr} dB")
        st.stop()

    # Show constellation
    st.markdown("### Signal Constellation")
    fig, ax = plt.subplots(figsize=(5, 5))
    I = samples[:, 0, :].flatten()
    Q = samples[:, 1, :].flatten()
    ax.scatter(I, Q, alpha=0.5, s=10, c="#2563eb")
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title(f"{modulation} @ {snr} dB")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    # Predictions from each model
    st.markdown("### Model Predictions")
    sample = samples[0]

    cols = st.columns(len(models))
    for i, (model_name, model) in enumerate(models.items()):
        with cols[i]:
            pred, probs = predict_modulation(model, sample)
            conf = probs[MODULATION_CLASSES.index(pred)] if pred else 0

            st.markdown(f"**{model_name}**")
            if pred == modulation:
                st.success(f"✓ {pred} ({conf:.0%})")
            else:
                st.error(f"✗ {pred} ({conf:.0%})")

            # Show top-3 predictions
            top3_idx = np.argsort(probs)[-3:][::-1]
            for idx in top3_idx:
                st.write(f"  {MODULATION_CLASSES[idx]}: {probs[idx]:.1%}")

elif comparison_type == "SNR Sweep":
    st.subheader("Accuracy vs SNR Comparison")

    if st.button("Run SNR Sweep (may take a moment)"):
        with st.spinner("Evaluating models across SNR levels..."):
            results = {}
            progress_bar = st.progress(0)

            for model_idx, (model_name, model) in enumerate(models.items()):
                accuracies = []

                for snr_idx, snr in enumerate(SNR_LEVELS):
                    # Get samples for this SNR
                    mask = test_snrs == snr
                    snr_data = test_data[mask]
                    snr_labels = test_labels[mask]

                    # Evaluate
                    correct = 0
                    total = 0
                    transform = Compose([PowerNormalize(), ToTensor()])

                    for sample, label in zip(snr_data[:200], snr_labels[:200]):  # Limit samples
                        x = transform(sample).unsqueeze(0)
                        with torch.no_grad():
                            logits = model(x)
                            pred = logits.argmax(dim=1).item()
                        if pred == label:
                            correct += 1
                        total += 1

                    accuracies.append(correct / total if total > 0 else 0)

                    # Update progress
                    progress = (model_idx * len(SNR_LEVELS) + snr_idx + 1) / (len(models) * len(SNR_LEVELS))
                    progress_bar.progress(progress)

                results[model_name] = accuracies

            progress_bar.empty()

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

            for i, (model_name, accs) in enumerate(results.items()):
                ax.plot(SNR_LEVELS, accs, "o-", label=model_name,
                        color=colors[i % len(colors)], linewidth=2, markersize=6)

            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Comparison: Accuracy vs SNR")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            plt.close(fig)

            # Summary table
            st.markdown("### Summary Statistics")
            summary_data = []
            for model_name, accs in results.items():
                high_snr_acc = np.mean([a for a, s in zip(accs, SNR_LEVELS) if s >= 10])
                low_snr_acc = np.mean([a for a, s in zip(accs, SNR_LEVELS) if s < 0])
                overall_acc = np.mean(accs)
                summary_data.append({
                    "Model": model_name,
                    "Overall": f"{overall_acc:.1%}",
                    "High SNR (≥10dB)": f"{high_snr_acc:.1%}",
                    "Low SNR (<0dB)": f"{low_snr_acc:.1%}",
                })
            st.table(summary_data)

elif comparison_type == "Impairment Robustness":
    st.subheader("Robustness Under Impairments")

    # Check if 2018 dataset is available
    has_2018 = "RadioML2018.01a" in get_available_datasets()

    impairment_options = ["CFO (Carrier Frequency Offset)", "I/Q Imbalance", "Rayleigh Fading", "Combined Impairments"]
    if has_2018:
        impairment_options.insert(0, "Cross-Dataset (2016→2018)")

    impairment_type = st.selectbox(
        "Impairment Type",
        impairment_options,
    )

    if st.button("Run Robustness Test"):
        with st.spinner("Testing robustness..."):
            results = {}
            combined_levels = []  # Will be populated for Combined Impairments

            if "Cross-Dataset" in impairment_type:
                st.markdown("""
                Testing models trained on **RadioML2016** against **RadioML2018** test data.
                Only the 8 overlapping classes are used.
                """)

                # Load 2018 dataset
                dataset_2018 = load_dataset(DATA_PATH_2018, "2018")

                if dataset_2018 is None:
                    st.error("Failed to load RadioML2018 dataset.")
                    st.stop()

                # Get 2018 test data
                test_2018_data, test_2018_labels, test_2018_snrs = dataset_2018["test"]
                class_names_2018 = dataset_2018["class_names"]

                # Build mapping from 2018 class indices to 2016 class indices
                class_2018_to_2016_idx = {}
                for cls_2018, cls_2016 in CLASS_NAME_MAPPING_2018_TO_2016.items():
                    if cls_2018 in class_names_2018 and cls_2016 in MODULATION_CLASSES:
                        idx_2018 = class_names_2018.index(cls_2018)
                        idx_2016 = MODULATION_CLASSES.index(cls_2016)
                        class_2018_to_2016_idx[idx_2018] = idx_2016

                # Filter 2018 data to overlapping classes
                overlapping_mask = np.isin(test_2018_labels, list(class_2018_to_2016_idx.keys()))
                filtered_2018_data = test_2018_data[overlapping_mask]
                filtered_2018_labels_orig = test_2018_labels[overlapping_mask]
                filtered_2018_snrs = test_2018_snrs[overlapping_mask]

                # Remap 2018 labels to 2016 indices
                filtered_2018_labels = np.array([class_2018_to_2016_idx[l] for l in filtered_2018_labels_orig])

                # Filter 2016 data to overlapping classes
                overlapping_2016_indices = [MODULATION_CLASSES.index(c) for c in OVERLAPPING_CLASSES if c in MODULATION_CLASSES]
                mask_2016 = np.isin(test_labels, overlapping_2016_indices)
                filtered_2016_data = test_data[mask_2016]
                filtered_2016_labels = test_labels[mask_2016]
                filtered_2016_snrs = test_snrs[mask_2016]

                # Subsample for speed (use high SNR samples)
                high_snr_mask_2016 = filtered_2016_snrs >= 0
                high_snr_mask_2018 = filtered_2018_snrs >= 0

                eval_2016_data = filtered_2016_data[high_snr_mask_2016][:500]
                eval_2016_labels = filtered_2016_labels[high_snr_mask_2016][:500]
                eval_2018_data = filtered_2018_data[high_snr_mask_2018][:500]
                eval_2018_labels = filtered_2018_labels[high_snr_mask_2018][:500]

                st.info(f"Evaluating on {len(eval_2016_labels)} samples from 2016 and {len(eval_2018_labels)} samples from 2018 (SNR ≥ 0 dB)")

                transform = Compose([PowerNormalize(), ToTensor()])

                # Evaluate each model on both datasets
                results_2016 = {}
                results_2018 = {}

                progress = st.progress(0)
                total_evals = len(models) * 2

                for i, (model_name, model) in enumerate(models.items()):
                    # Evaluate on 2016
                    correct = 0
                    for sample, label in zip(eval_2016_data, eval_2016_labels):
                        x = transform(sample).unsqueeze(0)
                        with torch.no_grad():
                            pred = model(x).argmax(dim=1).item()
                        if pred == label:
                            correct += 1
                    results_2016[model_name] = correct / len(eval_2016_labels)
                    progress.progress((i * 2 + 1) / total_evals)

                    # Evaluate on 2018
                    correct = 0
                    for sample, label in zip(eval_2018_data, eval_2018_labels):
                        x = transform(sample).unsqueeze(0)
                        with torch.no_grad():
                            pred = model(x).argmax(dim=1).item()
                        if pred == label:
                            correct += 1
                    results_2018[model_name] = correct / len(eval_2018_labels)
                    progress.progress((i * 2 + 2) / total_evals)

                progress.empty()

                # Plot grouped bar chart
                fig, ax = plt.subplots(figsize=(10, 6))

                x = np.arange(len(models))
                width = 0.35

                bars1 = ax.bar(x - width/2, [results_2016[m] for m in models.keys()],
                              width, label='Same Domain (2016→2016)', color='#22c55e', alpha=0.85)
                bars2 = ax.bar(x + width/2, [results_2018[m] for m in models.keys()],
                              width, label='Cross-Domain (2016→2018)', color='#dc2626', alpha=0.85)

                # Add value labels
                for bar in bars1:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{bar.get_height():.1%}', ha='center', fontsize=10)
                for bar in bars2:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{bar.get_height():.1%}', ha='center', fontsize=10)

                ax.set_ylabel('Accuracy')
                ax.set_title('Cross-Dataset Generalization (Overlapping Classes, SNR ≥ 0 dB)')
                ax.set_xticks(x)
                ax.set_xticklabels(list(models.keys()), rotation=15, ha='right')
                ax.legend(loc='lower right')
                ax.set_ylim(0, 1.15)
                ax.grid(True, alpha=0.2, axis='y')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Summary table
                st.markdown("### Cross-Dataset Performance Summary")
                summary_data = []
                for model_name in models.keys():
                    acc_2016 = results_2016[model_name]
                    acc_2018 = results_2018[model_name]
                    drop = acc_2016 - acc_2018
                    summary_data.append({
                        "Model": model_name,
                        "2016 (Same)": f"{acc_2016:.1%}",
                        "2018 (Cross)": f"{acc_2018:.1%}",
                        "Drop": f"{drop:+.1%}" if drop >= 0 else f"{drop:.1%}",
                    })
                st.table(summary_data)

                # Find best model for cross-dataset
                best_cross = max(results_2018.items(), key=lambda x: x[1])
                least_drop = min(models.keys(), key=lambda m: results_2016[m] - results_2018[m])

                st.success(f"**Best cross-dataset accuracy:** {best_cross[0]} ({best_cross[1]:.1%})")
                st.info(f"**Smallest accuracy drop:** {least_drop} ({results_2016[least_drop] - results_2018[least_drop]:.1%} drop)")

            elif "CFO" in impairment_type:
                cfo_values = [0, 500, 1000, 2000, 3000, 5000]
                x_label = "CFO (Hz)"
                x_values = cfo_values

                for model_name, model in models.items():
                    accuracies = []
                    for cfo in cfo_values:
                        correct = 0
                        total = 0
                        transform = Compose([PowerNormalize(), ToTensor()])

                        # Test on high SNR samples
                        mask = test_snrs >= 10
                        for sample, label in zip(test_data[mask][:200], test_labels[mask][:200]):
                            # Apply CFO
                            if cfo > 0:
                                cfo_transform = CarrierFrequencyOffset(delta_f=cfo, sample_rate=1e6)
                                sample = cfo_transform(sample)

                            x = transform(sample).unsqueeze(0)
                            with torch.no_grad():
                                pred = model(x).argmax(dim=1).item()
                            if pred == label:
                                correct += 1
                            total += 1
                        accuracies.append(correct / total)
                    results[model_name] = accuracies

            elif "I/Q" in impairment_type:
                iq_values = [0, 1, 2, 3, 4, 5]
                x_label = "I/Q Imbalance (dB)"
                x_values = iq_values

                for model_name, model in models.items():
                    accuracies = []
                    for iq in iq_values:
                        correct = 0
                        total = 0
                        transform = Compose([PowerNormalize(), ToTensor()])

                        mask = test_snrs >= 10
                        for sample, label in zip(test_data[mask][:200], test_labels[mask][:200]):
                            if iq > 0:
                                iq_transform = IQImbalance(amplitude_imbalance_db=iq, phase_imbalance_deg=iq)
                                sample = iq_transform(sample)

                            x = transform(sample).unsqueeze(0)
                            with torch.no_grad():
                                pred = model(x).argmax(dim=1).item()
                            if pred == label:
                                correct += 1
                            total += 1
                        accuracies.append(correct / total)
                    results[model_name] = accuracies

            elif "Rayleigh" in impairment_type:
                x_values = ["No Fading", "Rayleigh"]
                x_label = "Channel"

                for model_name, model in models.items():
                    accuracies = []
                    for fading in [False, True]:
                        correct = 0
                        total = 0
                        transform = Compose([PowerNormalize(), ToTensor()])

                        mask = test_snrs >= 10
                        for i, (sample, label) in enumerate(zip(test_data[mask][:200], test_labels[mask][:200])):
                            if fading:
                                fading_channel = RayleighFading(seed=i)
                                sample = fading_channel(sample)

                            x = transform(sample).unsqueeze(0)
                            with torch.no_grad():
                                pred = model(x).argmax(dim=1).item()
                            if pred == label:
                                correct += 1
                            total += 1
                        accuracies.append(correct / total)
                    results[model_name] = accuracies

            else:  # Combined Impairments
                levels = [
                    {"name": "Clean", "cfo": 0, "iq_amp": 0, "iq_phase": 0, "dc": 0},
                    {"name": "Mild", "cfo": 500, "iq_amp": 0.5, "iq_phase": 2, "dc": 0.05},
                    {"name": "Moderate", "cfo": 1000, "iq_amp": 1.0, "iq_phase": 5, "dc": 0.1},
                    {"name": "Severe", "cfo": 2000, "iq_amp": 2.0, "iq_phase": 10, "dc": 0.2},
                ]
                x_values = [level["name"] for level in levels]
                x_label = "Impairment Level"

                for model_name, model in models.items():
                    accuracies = []
                    for level in levels:
                        correct = 0
                        total = 0

                        mask = test_snrs >= 10
                        for sample, label in zip(test_data[mask][:200], test_labels[mask][:200]):
                            # Copy sample to avoid modifying original data
                            s = sample.copy()

                            # Apply combined impairments
                            if level["cfo"] > 0:
                                cfo_t = CarrierFrequencyOffset(
                                    delta_f=level["cfo"], sample_rate=1e6
                                )
                                s = cfo_t(s)
                            if level["iq_amp"] > 0 or level["iq_phase"] > 0:
                                iq_t = IQImbalance(
                                    amplitude_imbalance_db=level["iq_amp"],
                                    phase_imbalance_deg=level["iq_phase"]
                                )
                                s = iq_t(s)
                            if level["dc"] > 0:
                                dc_t = DCOffset(
                                    dc_i=level["dc"], dc_q=level["dc"], relative=True
                                )
                                s = dc_t(s)

                            transform = Compose([PowerNormalize(), ToTensor()])
                            x = transform(s).unsqueeze(0)
                            with torch.no_grad():
                                pred = model(x).argmax(dim=1).item()
                            if pred == label:
                                correct += 1
                            total += 1
                        accuracies.append(correct / total)
                    results[model_name] = accuracies

                # Store levels for display later
                combined_levels = levels

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

            if isinstance(x_values[0], str):
                x_plot = range(len(x_values))
                for i, (model_name, accs) in enumerate(results.items()):
                    ax.bar([x + i * 0.25 for x in x_plot], accs, 0.25,
                           label=model_name, color=colors[i % len(colors)])
                ax.set_xticks([x + 0.125 * (len(models) - 1) for x in x_plot])
                ax.set_xticklabels(x_values)
            else:
                for i, (model_name, accs) in enumerate(results.items()):
                    ax.plot(x_values, accs, "o-", label=model_name,
                            color=colors[i % len(colors)], linewidth=2, markersize=6)

            ax.set_xlabel(x_label)
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Robustness: {impairment_type}")
            if "CFO" in impairment_type or "I/Q" in impairment_type:
                ax.legend(loc="lower left")
            else:
                ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            plt.close(fig)

            # Show accuracy drop
            st.markdown("### Accuracy Degradation")
            for model_name, accs in results.items():
                baseline = accs[0]
                worst = min(accs)
                drop = baseline - worst
                st.write(f"**{model_name}**: {baseline:.1%} → {worst:.1%} (drop: {drop:.1%})")

            # Show impairment level details for combined impairments
            if "Combined" in impairment_type:
                st.markdown("### Impairment Levels")
                table_data = {
                    "Level": [lvl["name"] for lvl in combined_levels],
                    "CFO (Hz)": [lvl["cfo"] for lvl in combined_levels],
                    "I/Q Amp (dB)": [lvl["iq_amp"] for lvl in combined_levels],
                    "I/Q Phase (°)": [lvl["iq_phase"] for lvl in combined_levels],
                    "DC Offset": [lvl["dc"] for lvl in combined_levels],
                }
                st.dataframe(table_data, hide_index=True)

# Information section
with st.expander("About the Models", expanded=False):
    st.markdown("""
    ### Model Architectures

    **PF-CNN Baseline**
    - Dual-branch CNN processing amplitude and phase
    - Supervised training with CrossEntropy loss
    - Standard approach, susceptible to domain shift

    **PF-CNN + MDA-DMC**
    - Same architecture as baseline
    - Trained with Multi-Domain Augmentation (AGN, RSC, SSC)
    - More robust to variations in SNR, phase, and amplitude

    **CLSR-AMC**
    - Contrastive Learning with Self-Reconstruction
    - Multi-task training: contrastive + reconstruction + classification
    - Learns robust representations through self-supervision
    """)