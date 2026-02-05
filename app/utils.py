"""Shared utilities for the Streamlit dashboard."""

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robust_amc.data import (
    Compose,
    PowerNormalize,
    FamilyMapper,
    TorchSigDataset,
    PanoradioDataset,
    get_torchsig_loaders,
    get_panoradio_loaders,
    generate_torchsig_data,
    load_torchsig_data,
    load_panoradio_data,
    PANORADIO_SNR_LEVELS,
)
from robust_amc.data.channels import RayleighFading, RicianFading
from robust_amc.data.impairments import (
    CarrierFrequencyOffset,
    DCOffset,
    IQImbalance,
    PhaseNoise,
)
from robust_amc.data.transforms import ToTensor
from robust_amc.models import CLSRAMC, PFCNN, create_clsr_amc, create_pfcnn

# Default paths
TORCHSIG_CACHE_DIR = Path("data/torchsig_train")
PANORADIO_DIR = Path("data/panoradio")

# Family names (shared between TorchSig and Panoradio)
TORCHSIG_FAMILIES = ["PSK", "FSK", "AM", "SSB", "QAM"]
PANORADIO_FAMILIES = ["PSK", "FSK", "AM", "SSB", "OTHER"]

# Default SNR levels for synthetic data
SNR_LEVELS = list(range(-12, 22, 2))

# Model checkpoint paths
MODEL_CHECKPOINTS = {
    "PF-CNN (TorchSig)": Path("checkpoints/pfcnn_torchsig/best_model.pt"),
    "PF-CNN + Augment": Path("checkpoints/pfcnn_augmented/best_model.pt"),
    "CLSR-AMC": Path("checkpoints/clsr_amc/best_model.pt"),
}

# Number of classes for each model
MODEL_NUM_CLASSES = {
    "PF-CNN (TorchSig)": 5,
    "PF-CNN + Augment": 5,
    "CLSR-AMC": 5,
}


def get_family_names(dataset: str = "torchsig") -> list[str]:
    """Get family names for a dataset."""
    if dataset.lower() == "panoradio":
        return PANORADIO_FAMILIES
    return TORCHSIG_FAMILIES


def get_available_models() -> list[str]:
    """Get list of models that have trained checkpoints available."""
    available = []
    for name, path in MODEL_CHECKPOINTS.items():
        if path.exists():
            available.append(name)
    return available


@st.cache_resource
def load_model_by_name(model_name: str) -> Optional[Union[PFCNN, CLSRAMC]]:
    """Load a model by name (cached).

    Args:
        model_name: Model name from MODEL_CHECKPOINTS

    Returns:
        Loaded model or None if not found
    """
    if model_name not in MODEL_CHECKPOINTS:
        st.error(f"Unknown model: {model_name}")
        return None

    checkpoint_path = MODEL_CHECKPOINTS[model_name]
    if not checkpoint_path.exists():
        st.warning(f"Model checkpoint not found at {checkpoint_path}")
        return None

    num_classes = MODEL_NUM_CLASSES.get(model_name, 5)

    # Create appropriate model architecture
    if "CLSR" in model_name:
        model = create_clsr_amc(num_classes=num_classes, variant="default")
    else:
        model = create_pfcnn(num_classes=num_classes, variant="default")

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model


@st.cache_resource
def load_model(checkpoint_path: Optional[Path] = None, num_classes: int = 5) -> Optional[PFCNN]:
    """Load a PF-CNN model from checkpoint (cached).

    For backwards compatibility.
    """
    if checkpoint_path is None:
        checkpoint_path = MODEL_CHECKPOINTS.get("PF-CNN (TorchSig)")
        if checkpoint_path is None:
            return None

    if not checkpoint_path.exists():
        st.warning(f"Model checkpoint not found at {checkpoint_path}")
        return None

    model = create_pfcnn(num_classes=num_classes, variant="default")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model


@st.cache_data
def load_dataset(dataset_name: str = "torchsig") -> Optional[dict]:
    """Load dataset (cached).

    Args:
        dataset_name: "torchsig" or "panoradio"

    Returns:
        Dictionary with data splits and metadata
    """
    if dataset_name.lower() == "torchsig":
        return _load_torchsig_dataset()
    elif dataset_name.lower() == "panoradio":
        return _load_panoradio_dataset()
    else:
        st.error(f"Unknown dataset: {dataset_name}")
        return None


def _load_torchsig_dataset() -> Optional[dict]:
    """Load TorchSig dataset."""
    if not TORCHSIG_CACHE_DIR.exists():
        st.info("Generating TorchSig data (first time only)...")
        try:
            generate_torchsig_data(
                TORCHSIG_CACHE_DIR,
                num_samples_per_class=2000,
                signal_length=1024,
            )
        except Exception as e:
            st.error(f"Failed to generate TorchSig data: {e}")
            return None

    try:
        data, labels, snrs = load_torchsig_data(TORCHSIG_CACHE_DIR)
        mapper = FamilyMapper(Path("configs/label_maps/torchsig_to_family.yaml"))

        # Map labels to family indices
        family_indices = np.array([mapper.get_family_idx(str(lbl)) or -1 for lbl in labels])
        valid_mask = family_indices >= 0
        data = data[valid_mask]
        labels = labels[valid_mask]
        snrs = snrs[valid_mask]
        family_indices = family_indices[valid_mask]

        # Simple split (60/20/20)
        n = len(data)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)

        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        return {
            "train": (data[train_idx], family_indices[train_idx], snrs[train_idx]),
            "val": (data[val_idx], family_indices[val_idx], snrs[val_idx]),
            "test": (data[test_idx], family_indices[test_idx], snrs[test_idx]),
            "raw_labels": labels,
            "family_names": mapper.family_names,
            "snr_levels": SNR_LEVELS,
            "dataset_name": "TorchSig",
        }
    except Exception as e:
        st.error(f"Failed to load TorchSig data: {e}")
        return None


def _load_panoradio_dataset() -> Optional[dict]:
    """Load Panoradio dataset."""
    if not PANORADIO_DIR.exists():
        st.warning(
            f"Panoradio data not found at {PANORADIO_DIR}. "
            "Download from: https://panoradio-sdr.de/radio-signal-classification-dataset/"
        )
        return None

    try:
        data, labels, snrs = load_panoradio_data(PANORADIO_DIR)
        mapper = FamilyMapper(Path("configs/label_maps/panoradio_to_family.yaml"))

        # Map labels to family indices
        family_indices = np.array([mapper.get_family_idx(str(lbl)) or -1 for lbl in labels])
        valid_mask = family_indices >= 0
        data = data[valid_mask]
        labels = labels[valid_mask]
        snrs = snrs[valid_mask]
        family_indices = family_indices[valid_mask]

        # Simple split (20/80 for zero-shot style eval)
        n = len(data)
        n_val = int(0.2 * n)

        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        test_idx = indices[n_val:]

        return {
            "val": (data[val_idx], family_indices[val_idx], snrs[val_idx]),
            "test": (data[test_idx], family_indices[test_idx], snrs[test_idx]),
            "raw_labels": labels,
            "family_names": mapper.family_names,
            "snr_levels": PANORADIO_SNR_LEVELS,
            "dataset_name": "Panoradio",
        }
    except Exception as e:
        st.error(f"Failed to load Panoradio data: {e}")
        return None


def get_available_datasets() -> list[str]:
    """Get list of available datasets."""
    available = []

    # TorchSig is always available (can be generated)
    available.append("TorchSig")

    # Panoradio requires data download
    if PANORADIO_DIR.exists() and (PANORADIO_DIR / "rscd_2048.npy").exists():
        available.append("Panoradio")

    return available


def normalize_signal(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize signal to unit power."""
    if np.iscomplexobj(signal):
        power = np.mean(np.abs(signal) ** 2)
        return signal / np.sqrt(power + eps)
    else:
        power = np.mean(signal[0] ** 2 + signal[1] ** 2)
        return signal / np.sqrt(power + eps)


def normalize_samples(samples: np.ndarray) -> np.ndarray:
    """Normalize a batch of samples to unit power."""
    return np.array([normalize_signal(s) for s in samples])


def complex_to_iq(signal: np.ndarray) -> np.ndarray:
    """Convert complex signal to I/Q format."""
    if np.iscomplexobj(signal):
        return np.stack([signal.real, signal.imag], axis=0).astype(np.float32)
    return signal


def get_samples_for_family(
    data: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    family_idx: int,
    snr: float,
    n_samples: int = 10,
    normalize: bool = True,
) -> Optional[np.ndarray]:
    """Get samples for a specific family and SNR.

    Args:
        data: Signal data array (can be complex or I/Q)
        labels: Family index array
        snrs: SNR array
        family_idx: Family index to filter by
        snr: SNR value to filter by
        n_samples: Number of samples to return
        normalize: Whether to normalize samples to unit power

    Returns:
        Array of samples or None if no matching samples found
    """
    # Find matching samples
    snr_tolerance = 1.0  # Allow 1 dB tolerance for SNR matching
    mask = (labels == family_idx) & (np.abs(snrs - snr) <= snr_tolerance)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None

    n = min(n_samples, len(indices))
    selected = np.random.choice(indices, size=n, replace=False)
    samples = data[selected]

    # Convert complex to I/Q if needed
    if np.iscomplexobj(samples):
        samples = np.stack([samples.real, samples.imag], axis=1).astype(np.float32)

    if normalize:
        samples = normalize_samples(samples)

    return samples


def apply_impairments(
    signal: np.ndarray,
    cfo_hz: float = 0,
    iq_amp_db: float = 0,
    iq_phase_deg: float = 0,
    dc_i: float = 0,
    dc_q: float = 0,
    phase_noise_std: float = 0,
    sample_rate: float = 1e6,
) -> np.ndarray:
    """Apply a chain of impairments to a signal."""
    result = signal.copy()

    # Convert complex to I/Q if needed
    if np.iscomplexobj(result):
        result = complex_to_iq(result)

    # Apply CFO
    if cfo_hz != 0:
        cfo = CarrierFrequencyOffset(delta_f=cfo_hz, sample_rate=sample_rate)
        result = cfo(result)

    # Apply I/Q imbalance
    if iq_amp_db != 0 or iq_phase_deg != 0:
        iq_imb = IQImbalance(
            amplitude_imbalance_db=iq_amp_db,
            phase_imbalance_deg=iq_phase_deg,
        )
        result = iq_imb(result)

    # Apply DC offset
    if dc_i != 0 or dc_q != 0:
        dc = DCOffset(dc_i=dc_i, dc_q=dc_q, relative=True)
        result = dc(result)

    # Apply phase noise
    if phase_noise_std > 0:
        pn = PhaseNoise(std_per_sample=phase_noise_std)
        result = pn(result)

    return result


def apply_fading(
    signal: np.ndarray,
    fading_type: str = "none",
    k_factor: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply fading channel to a signal."""
    if fading_type == "none":
        return signal

    # Convert complex to I/Q if needed
    if np.iscomplexobj(signal):
        signal = complex_to_iq(signal)

    if fading_type == "rayleigh":
        channel = RayleighFading(seed=seed)
    elif fading_type == "rician":
        channel = RicianFading(k_factor=k_factor, seed=seed)
    else:
        return signal

    return channel(signal)


def predict_family(
    model: nn.Module,
    signal: np.ndarray,
    family_names: list[str],
    normalize: bool = True,
    crop_length: int = 128,
) -> tuple[Optional[str], Optional[np.ndarray]]:
    """Run inference on a signal and return family prediction.

    Args:
        model: Trained model
        signal: Input signal (complex or I/Q format)
        family_names: List of family names
        normalize: Whether to normalize the signal
        crop_length: Crop signal to this length

    Returns:
        Tuple of (predicted_family_name, probabilities)
    """
    if model is None:
        return None, None

    # Convert complex to I/Q if needed
    if np.iscomplexobj(signal):
        signal = complex_to_iq(signal)

    # Crop to expected length
    if signal.shape[1] > crop_length:
        start = (signal.shape[1] - crop_length) // 2
        signal = signal[:, start : start + crop_length]
    elif signal.shape[1] < crop_length:
        pad = crop_length - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad)), mode="constant")

    # Prepare input
    if normalize:
        transform = Compose([PowerNormalize(), ToTensor()])
        x = transform(signal)
    else:
        x = torch.from_numpy(signal).float()

    # Add batch dimension
    x = x.unsqueeze(0)

    # Move input to same device as model
    device = next(model.parameters()).device
    x = x.to(device)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = logits.argmax(dim=1).item()

    if pred_idx < len(family_names):
        return family_names[pred_idx], probs
    return None, probs


# Backwards compatibility alias
def predict_modulation(
    model: nn.Module,
    signal: np.ndarray,
    normalize: bool = True,
) -> tuple[Optional[str], Optional[np.ndarray]]:
    """Backwards compatible prediction function."""
    return predict_family(model, signal, TORCHSIG_FAMILIES, normalize)
