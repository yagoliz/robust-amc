#!/usr/bin/env python3
"""Cross-dataset evaluation for domain shift analysis.

Evaluates models trained on one RadioML dataset against the other
to measure robustness to domain shift.

Usage:
    # Evaluate 2016-trained models on 2018 data
    uv run python scripts/evaluate_cross_dataset.py --train-dataset 2016 --eval-dataset 2018

    # Evaluate 2018-trained models on 2016 data
    uv run python scripts/evaluate_cross_dataset.py --train-dataset 2018 --eval-dataset 2016
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_amc.data import (
    CLASS_NAME_MAPPING_2018_TO_2016,
    OVERLAPPING_CLASSES,
    Compose,
    PowerNormalize,
    load_radioml2016a,
    load_radioml2018a,
)
from robust_amc.data.radioml2018_loader import MODULATION_CLASSES_2018 as CLASSES_2018
from robust_amc.data.radioml_loader import MODULATION_CLASSES as CLASSES_2016
from robust_amc.data.transforms import ToTensor
from robust_amc.evaluation import accuracy_by_snr
from robust_amc.models import create_clsr_amc, create_pfcnn
from robust_amc.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument(
        "--train-dataset",
        type=str,
        choices=["2016", "2018"],
        required=True,
        help="Dataset the models were trained on",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        choices=["2016", "2018"],
        required=True,
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-path-2016",
        type=Path,
        default=Path("data/RML2016.10a_dict.pkl"),
        help="Path to RadioML2016.10a dataset",
    )
    parser.add_argument(
        "--data-path-2018",
        type=Path,
        default=Path("data/GOLD_XYZ_OSC.0001_1024.hdf5"),
        help="Path to RadioML2018.01a dataset",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for faster testing)",
    )
    return parser.parse_args()


def load_eval_data(
    dataset: str,
    data_path_2016: Path,
    data_path_2018: Path,
    max_samples: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load evaluation dataset with overlapping classes only.

    Returns data with class indices mapped to OVERLAPPING_CLASSES order.
    """
    if dataset == "2016":
        if not data_path_2016.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path_2016}")
        data, labels, snrs = load_radioml2016a(data_path_2016)

        # Filter to overlapping classes and remap indices
        overlap_indices_2016 = [CLASSES_2016.index(c) for c in OVERLAPPING_CLASSES if c in CLASSES_2016]
        mask = np.isin(labels, overlap_indices_2016)
        data = data[mask]
        labels_orig = labels[mask]
        snrs = snrs[mask]

        # Remap to OVERLAPPING_CLASSES order
        old_to_new = {CLASSES_2016.index(c): OVERLAPPING_CLASSES.index(c)
                      for c in OVERLAPPING_CLASSES if c in CLASSES_2016}
        labels = np.array([old_to_new[l] for l in labels_orig])

    else:  # 2018
        if not data_path_2018.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path_2018}")
        data, labels, snrs, class_names = load_radioml2018a(
            data_path_2018,
            split_segments=True,
            overlapping_only=True,
        )

        # Map 2018 class names to OVERLAPPING_CLASSES order
        # The loader already filtered, but we need to remap to consistent indices
        name_2018_to_overlap = {}
        for name_2018, name_2016 in CLASS_NAME_MAPPING_2018_TO_2016.items():
            if name_2016 in OVERLAPPING_CLASSES:
                name_2018_to_overlap[name_2018] = OVERLAPPING_CLASSES.index(name_2016)

        old_to_new = {i: name_2018_to_overlap[class_names[i]]
                      for i in range(len(class_names))
                      if class_names[i] in name_2018_to_overlap}

        # Filter to classes we can map
        valid_old_indices = list(old_to_new.keys())
        mask = np.isin(labels, valid_old_indices)
        data = data[mask]
        labels_orig = labels[mask]
        snrs = snrs[mask]
        labels = np.array([old_to_new[l] for l in labels_orig])

    # Limit samples if requested
    if max_samples is not None and len(data) > max_samples:
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data[indices]
        labels = labels[indices]
        snrs = snrs[indices]

    return data, labels, snrs, OVERLAPPING_CLASSES


def evaluate_model(
    model: torch.nn.Module,
    data: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    device: str,
    batch_size: int = 256,
    train_class_to_overlap: dict = None,
) -> dict:
    """Evaluate model and return metrics.

    Args:
        model: The trained model
        data: Input data
        labels: Target labels (in overlapping class indices)
        snrs: SNR values
        device: Device to use
        batch_size: Batch size
        train_class_to_overlap: Mapping from training class indices to overlapping class indices
    """
    model = model.to(device)
    model.eval()

    transform = Compose([PowerNormalize(), ToTensor()])

    all_preds = []
    all_targets = []
    all_snrs = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            batch_snrs = snrs[i:i + batch_size]

            # Transform batch
            x_batch = torch.stack([transform(x) for x in batch_data]).to(device)

            # Forward pass
            logits = model(x_batch)

            if train_class_to_overlap is not None:
                # Map model output to overlapping class indices
                # Only consider logits for overlapping classes
                overlap_indices = sorted(train_class_to_overlap.keys())
                logits_overlap = logits[:, overlap_indices]
                preds_in_overlap = logits_overlap.argmax(dim=1).cpu().numpy()
                # Map back to the overlap class index
                preds = np.array([train_class_to_overlap[overlap_indices[p]] for p in preds_in_overlap])
            else:
                preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(batch_labels)
            all_snrs.extend(batch_snrs)

    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    snrs_arr = np.array(all_snrs)

    # Compute metrics
    accuracy = (predictions == targets).mean()
    snr_acc = accuracy_by_snr(targets, predictions, snrs_arr)

    return {
        "accuracy": float(accuracy),
        "snr_accuracy": snr_acc,
        "predictions": predictions,
        "targets": targets,
        "snrs": snrs_arr,
    }


def main():
    args = parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    print("=" * 60)
    print(f"Cross-Dataset Evaluation: Train on {args.train_dataset}, Eval on {args.eval_dataset}")
    print("=" * 60)

    # Load evaluation data
    print(f"\n1. Loading {args.eval_dataset} dataset for evaluation...")
    try:
        data, labels, snrs, class_names = load_eval_data(
            args.eval_dataset,
            args.data_path_2016,
            args.data_path_2018,
            args.max_samples,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"   Samples: {len(data)}")
    print(f"   Classes: {len(class_names)} ({', '.join(class_names[:5])}...)")
    print(f"   SNR range: {snrs.min():.0f} to {snrs.max():.0f} dB")

    # Find available models trained on the specified dataset
    models_to_evaluate = []

    # Check for baseline
    baseline_path = args.checkpoints_dir / f"baseline_{args.train_dataset}" / "best_model.pt"
    if not baseline_path.exists():
        # Try legacy path without dataset suffix
        baseline_path = args.checkpoints_dir / "baseline" / "best_model.pt"
    if baseline_path.exists():
        models_to_evaluate.append(("Baseline", baseline_path, "pfcnn"))

    # Check for MDA-DMC
    mda_path = args.checkpoints_dir / f"mda_dmc_{args.train_dataset}" / "best_model.pt"
    if not mda_path.exists():
        mda_path = args.checkpoints_dir / "mda_dmc" / "best_model.pt"
    if mda_path.exists():
        models_to_evaluate.append(("MDA-DMC", mda_path, "pfcnn"))

    # Check for CLSR-AMC
    clsr_path = args.checkpoints_dir / f"clsr_amc_{args.train_dataset}" / "best_model.pt"
    if not clsr_path.exists():
        clsr_path = args.checkpoints_dir / "clsr_amc" / "best_model.pt"
    if clsr_path.exists():
        models_to_evaluate.append(("CLSR-AMC", clsr_path, "clsr_amc"))

    if not models_to_evaluate:
        print(f"\nNo models found trained on {args.train_dataset} dataset!")
        print("Available checkpoints should be in:")
        print(f"  {args.checkpoints_dir}/baseline_{args.train_dataset}/")
        print(f"  {args.checkpoints_dir}/mda_dmc_{args.train_dataset}/")
        print(f"  {args.checkpoints_dir}/clsr_amc_{args.train_dataset}/")
        sys.exit(1)

    print(f"\n2. Found {len(models_to_evaluate)} models to evaluate:")
    for name, path, _ in models_to_evaluate:
        print(f"   - {name}: {path}")

    # Build mapping from training dataset class indices to overlapping class indices
    if args.train_dataset == "2016":
        train_classes = CLASSES_2016
    else:
        train_classes = CLASSES_2018

    # Map: training class index -> overlapping class index (if class is in overlap)
    train_class_to_overlap = {}
    for train_idx, train_name in enumerate(train_classes):
        # Check if this training class maps to an overlapping class
        if args.train_dataset == "2016":
            # 2016 names are used directly in OVERLAPPING_CLASSES
            if train_name in OVERLAPPING_CLASSES:
                overlap_idx = OVERLAPPING_CLASSES.index(train_name)
                train_class_to_overlap[train_idx] = overlap_idx
        else:
            # 2018 names need to be mapped via CLASS_NAME_MAPPING_2018_TO_2016
            if train_name in CLASS_NAME_MAPPING_2018_TO_2016:
                name_2016 = CLASS_NAME_MAPPING_2018_TO_2016[train_name]
                if name_2016 in OVERLAPPING_CLASSES:
                    overlap_idx = OVERLAPPING_CLASSES.index(name_2016)
                    train_class_to_overlap[train_idx] = overlap_idx

    print(f"\n   Class mapping ({len(train_class_to_overlap)} classes):")
    for train_idx, overlap_idx in sorted(train_class_to_overlap.items()):
        print(f"      {train_classes[train_idx]} -> {OVERLAPPING_CLASSES[overlap_idx]}")

    # Evaluate each model
    results = {}
    snr_accuracies = {}

    print(f"\n3. Evaluating on {args.eval_dataset} dataset...")
    for name, checkpoint_path, model_type in models_to_evaluate:
        print(f"\n   {name}:")

        # Create model with ORIGINAL number of classes (not overlapping)
        num_train_classes = len(train_classes)
        if model_type == "clsr_amc":
            model = create_clsr_amc(num_classes=num_train_classes)
        else:
            model = create_pfcnn(num_classes=num_train_classes)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate with class mapping
        result = evaluate_model(
            model, data, labels, snrs, device,
            train_class_to_overlap=train_class_to_overlap,
        )
        results[name] = result
        snr_accuracies[name] = [result["snr_accuracy"][snr] for snr in sorted(result["snr_accuracy"].keys())]

        print(f"      Overall Accuracy: {result['accuracy']:.4f}")

    # Get SNR values for plotting
    snr_values = sorted(results[list(results.keys())[0]]["snr_accuracy"].keys())

    # Print SNR accuracy table
    print(f"\n4. Accuracy by SNR:")
    print(f"   {'SNR':>6} | " + " | ".join(f"{name:>10}" for name in results.keys()))
    print("   " + "-" * (10 + 14 * len(results)))
    for snr in snr_values:
        row = f"   {snr:>4} dB |"
        for name in results.keys():
            acc = results[name]["snr_accuracy"].get(snr, 0)
            row += f" {acc:>10.4f} |"
        print(row)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    for i, (name, accs) in enumerate(snr_accuracies.items()):
        ax.plot(snr_values, accs, marker="o", label=name, linewidth=2, color=colors[i])

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Cross-Dataset: Train on {args.train_dataset}, Eval on {args.eval_dataset}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    save_path = args.results_dir / f"cross_dataset_{args.train_dataset}_to_{args.eval_dataset}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   Plot saved: {save_path}")

    # Save results JSON
    results_data = {
        "train_dataset": args.train_dataset,
        "eval_dataset": args.eval_dataset,
        "n_samples": len(data),
        "n_classes": len(class_names),
        "class_names": class_names,
        "models": {
            name: {
                "accuracy": results[name]["accuracy"],
                "snr_accuracy": {str(k): float(v) for k, v in results[name]["snr_accuracy"].items()},
            }
            for name in results.keys()
        },
    }

    json_path = args.results_dir / f"cross_dataset_{args.train_dataset}_to_{args.eval_dataset}.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"   Results saved: {json_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'Overall Acc':>12} {'High SNR (≥10)':>15}")
    print("-" * 44)
    for name in results.keys():
        overall = results[name]["accuracy"]
        high_snr_accs = [results[name]["snr_accuracy"][s] for s in snr_values if s >= 10]
        high_snr = np.mean(high_snr_accs) if high_snr_accs else 0
        print(f"{name:<15} {overall:>12.4f} {high_snr:>15.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()