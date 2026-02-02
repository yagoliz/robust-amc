"""Shared utilities for the Streamlit dashboard."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robust_amc.data import (
    load_radioml2016a,
    stratified_split,
    RadioMLDataset,
    PowerNormalize,
    Compose,
)
from robust_amc.data.transforms import ToTensor
from robust_amc.data.radioml_loader import MODULATION_CLASSES, SNR_LEVELS
from robust_amc.data.channels import AWGN, RayleighFading, RicianFading
from robust_amc.data.impairments import (
    CarrierFrequencyOffset,
    IQImbalance,
    DCOffset,
    PhaseNoise,
)
from robust_amc.models import PFCNN


# Default paths
DATA_PATH = Path("data/RML2016.10a_dict.pkl")
CHECKPOINT_PATH = Path("checkpoints/baseline/best_model.pt")


@st.cache_resource
def load_model(checkpoint_path: Path = CHECKPOINT_PATH) -> PFCNN:
    """Load the trained model (cached)."""
    model = PFCNN(num_classes=len(MODULATION_CLASSES))

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    else:
        st.warning(f"Model checkpoint not found at {checkpoint_path}")
        return None


@st.cache_data
def load_dataset(data_path: Path = DATA_PATH) -> dict:
    """Load and split the dataset (cached)."""
    if not data_path.exists():
        return None

    data, labels, snrs = load_radioml2016a(data_path)
    splits = stratified_split(data, labels, snrs)

    return {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "class_names": MODULATION_CLASSES,
        "snr_levels": SNR_LEVELS,
    }


def normalize_signal(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize signal to unit power."""
    power = np.mean(signal[0] ** 2 + signal[1] ** 2)
    return signal / np.sqrt(power + eps)


def normalize_samples(samples: np.ndarray) -> np.ndarray:
    """Normalize a batch of samples to unit power."""
    return np.array([normalize_signal(s) for s in samples])


def get_samples_for_modulation(
    data: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    modulation: str,
    snr: int,
    n_samples: int = 10,
    normalize: bool = True,
) -> np.ndarray:
    """Get samples for a specific modulation and SNR."""
    mod_idx = MODULATION_CLASSES.index(modulation)
    mask = (labels == mod_idx) & (snrs == snr)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None

    n = min(n_samples, len(indices))
    selected = np.random.choice(indices, size=n, replace=False)
    samples = data[selected]

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
    seed: int = None,
) -> np.ndarray:
    """Apply fading channel to a signal."""
    if fading_type == "none":
        return signal

    if fading_type == "rayleigh":
        channel = RayleighFading(seed=seed)
    elif fading_type == "rician":
        channel = RicianFading(k_factor=k_factor, seed=seed)
    else:
        return signal

    return channel(signal)


def predict_modulation(
    model: PFCNN,
    signal: np.ndarray,
    normalize: bool = True,
) -> tuple[str, np.ndarray]:
    """Run inference on a signal and return prediction.

    Returns:
        Tuple of (predicted_class_name, probabilities)
    """
    if model is None:
        return None, None

    # Prepare input
    if normalize:
        transform = Compose([PowerNormalize(), ToTensor()])
        x = transform(signal)
    else:
        x = torch.from_numpy(signal).float()

    # Add batch dimension
    x = x.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        pred_idx = logits.argmax(dim=1).item()

    return MODULATION_CLASSES[pred_idx], probs


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"