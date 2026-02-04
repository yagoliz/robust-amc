"""Shared utilities for the Streamlit dashboard."""

import sys
from pathlib import Path
from typing import Union

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robust_amc.data import (
    OVERLAPPING_CLASSES,
    PREPROCESSED_2018_DIR,
    Compose,
    PowerNormalize,
    is_preprocessed_available,
    load_radioml2016a,
    load_radioml2018a,
    stratified_split,
    stratified_split_2018,
)
from robust_amc.data.channels import RayleighFading, RicianFading
from robust_amc.data.impairments import (
    CarrierFrequencyOffset,
    DCOffset,
    IQImbalance,
    PhaseNoise,
)
from robust_amc.data.radioml2018_loader import MODULATION_CLASSES_2018, SNR_LEVELS_2018
from robust_amc.data.radioml_loader import MODULATION_CLASSES, SNR_LEVELS
from robust_amc.data.transforms import ToTensor
from robust_amc.models import CLSRAMC, PFCNN, create_clsr_amc

# Default paths
DATA_PATH_2016 = Path("data/RML2016.10a_dict.pkl")
DATA_PATH_2018 = Path("data/GOLD_XYZ_OSC.0001_1024.hdf5")

# For backwards compatibility
DATA_PATH = DATA_PATH_2016

# Model checkpoint paths
MODEL_CHECKPOINTS = {
    "PF-CNN Baseline": Path("checkpoints/baseline/best_model.pt"),
    "PF-CNN + MDA-DMC": Path("checkpoints/mda_dmc/best_model.pt"),
    "CLSR-AMC": Path("checkpoints/clsr_amc/best_model.pt"),
    "PF-CNN Baseline (2018)": Path("checkpoints/baseline_2018/best_model.pt"),
}

# Models trained on RadioML2018 (24 classes)
MODELS_2018 = {"PF-CNN Baseline (2018)"}


def is_model_2018(model_name: str) -> bool:
    """Check if a model was trained on RadioML2018."""
    return model_name in MODELS_2018


def get_model_class_names(model_name: str) -> list[str]:
    """Get the class names for a model based on its training dataset."""
    if model_name in MODELS_2018:
        return MODULATION_CLASSES_2018
    return MODULATION_CLASSES


def get_available_models() -> list[str]:
    """Get list of models that have trained checkpoints available."""
    available = []
    for name, path in MODEL_CHECKPOINTS.items():
        if path.exists():
            available.append(name)
    return available


@st.cache_resource
def load_model_by_name(model_name: str) -> Union[PFCNN, CLSRAMC, None]:
    """Load a model by name (cached).

    Args:
        model_name: One of "PF-CNN Baseline", "PF-CNN + MDA-DMC", "CLSR-AMC",
                   or "PF-CNN Baseline (2018)"

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

    # Determine number of classes based on model
    if model_name in MODELS_2018:
        num_classes = len(MODULATION_CLASSES_2018)
    else:
        num_classes = len(MODULATION_CLASSES)

    # Create appropriate model architecture
    if model_name == "CLSR-AMC":
        model = create_clsr_amc(num_classes=num_classes, variant="default")
    else:
        # Baseline and MDA-DMC use PF-CNN architecture
        model = PFCNN(num_classes=num_classes)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    return model


@st.cache_resource
def load_model(checkpoint_path: Path = None) -> PFCNN:
    """Load the baseline trained model (cached).

    For backwards compatibility - loads baseline model by default.
    """
    if checkpoint_path is None:
        checkpoint_path = MODEL_CHECKPOINTS["PF-CNN Baseline"]

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


@st.cache_resource
def _load_preprocessed_2018_cached() -> dict:
    """Load preprocessed 2018 dataset with resource caching.

    Uses @st.cache_resource instead of @st.cache_data because the
    memory-mapped arrays cannot be serialized.
    """
    return _load_preprocessed_2018()


@st.cache_data
def load_dataset(data_path: Path = DATA_PATH, dataset_version: str = "2016") -> dict:
    """Load and split the dataset (cached).

    Args:
        data_path: Path to dataset file
        dataset_version: "2016" or "2018"

    Returns:
        Dictionary with train/val/test splits and metadata
    """
    if dataset_version == "2018":
        # Prefer preprocessed format if available (much faster)
        if is_preprocessed_available():
            # Use resource caching for memory-mapped arrays
            return _load_preprocessed_2018_cached()

        # Fall back to HDF5 (slow but works)
        if not data_path.exists():
            return None

        st.warning(
            "Using raw HDF5 format (slow). "
            "Run `uv run python scripts/preprocess_radioml2018.py` for faster loading."
        )
        data, labels, snrs, class_names = load_radioml2018a(
            data_path,
            split_segments=True,
            overlapping_only=False,
        )
        splits = stratified_split_2018(data, labels, snrs)
        return {
            "train": splits["train"],
            "val": splits["val"],
            "test": splits["test"],
            "class_names": class_names,
            "snr_levels": SNR_LEVELS_2018,
            "dataset_version": "2018",
        }
    else:
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
            "dataset_version": "2016",
        }


class _MappedSplitView:
    """View into memory-mapped arrays for a specific split.

    Keeps the underlying data memory-mapped. Labels and SNRs are loaded
    eagerly (they're small), but signal data is only loaded on-demand
    when specific samples are requested.
    """

    def __init__(self, data_mmap, labels_mmap, snrs_mmap, indices):
        self._data_mmap = data_mmap  # Keep memory-mapped
        self._indices = indices
        # Labels and SNRs are small - load them for fast filtering
        self._labels = np.array(labels_mmap[indices])
        self._snrs = np.array(snrs_mmap[indices])

    def __getitem__(self, idx):
        """Support tuple unpacking and indexing."""
        if isinstance(idx, int):
            if idx == 0:
                return self  # Return self for data - it handles __getitem__
            elif idx == 1:
                return self._labels
            elif idx == 2:
                return self._snrs
            raise IndexError(f"Index {idx} out of range")
        # Array indexing - load only requested samples from mmap
        real_indices = self._indices[idx]
        return np.array(self._data_mmap[real_indices])

    def __iter__(self):
        """Support tuple unpacking: data, labels, snrs = split."""
        yield self  # data (this object handles array access)
        yield self._labels
        yield self._snrs

    def __len__(self):
        return 3

    @property
    def shape(self):
        return (len(self._indices), 2, self._data_mmap.shape[2])


class _MappedDataProxy:
    """Proxy for data array that loads from mmap on access."""

    def __init__(self, data_mmap, indices):
        self._data_mmap = data_mmap
        self._indices = indices
        self._shape = (len(indices), 2, data_mmap.shape[2])

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, idx):
        """Load only requested samples from memory-mapped array."""
        if isinstance(idx, (int, np.integer)):
            # Single sample
            real_idx = self._indices[idx]
            return np.array(self._data_mmap[real_idx])
        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            # Boolean mask - convert to integer indices
            int_indices = np.where(idx)[0]
            real_indices = self._indices[int_indices]
            return np.array(self._data_mmap[real_indices])
        elif isinstance(idx, (list, np.ndarray)):
            # Multiple samples - load only these
            real_indices = self._indices[idx]
            return np.array(self._data_mmap[real_indices])
        elif isinstance(idx, slice):
            # Slice - convert to indices
            slice_indices = range(*idx.indices(len(self._indices)))
            real_indices = self._indices[list(slice_indices)]
            return np.array(self._data_mmap[real_indices])
        raise TypeError(f"Invalid index type: {type(idx)}")

    def __len__(self):
        return len(self._indices)


def _load_preprocessed_2018() -> dict:
    """Load preprocessed RadioML2018 dataset efficiently.

    Uses memory-mapped arrays - signal data is only loaded when specific
    samples are accessed, not at initial load time. Labels and SNRs are
    loaded eagerly since they're small and needed for filtering.
    """
    import json

    preprocessed_dir = PREPROCESSED_2018_DIR

    # Load memory-mapped arrays (instant, no data loaded yet)
    data = np.load(preprocessed_dir / "data.npy", mmap_mode="r")
    labels = np.load(preprocessed_dir / "labels.npy", mmap_mode="r")
    snrs = np.load(preprocessed_dir / "snrs.npy", mmap_mode="r")

    # Load metadata (small, loads instantly)
    with open(preprocessed_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load pre-computed split indices (small, loads instantly)
    train_idx = np.load(preprocessed_dir / "indices" / "train.npy")
    val_idx = np.load(preprocessed_dir / "indices" / "val.npy")
    test_idx = np.load(preprocessed_dir / "indices" / "test.npy")

    # Create proxy objects that load data on-demand
    def make_split(indices):
        return (
            _MappedDataProxy(data, indices),
            np.array(labels[indices]),  # Small, load eagerly
            np.array(snrs[indices]),    # Small, load eagerly
        )

    return {
        "train": make_split(train_idx),
        "val": make_split(val_idx),
        "test": make_split(test_idx),
        "class_names": metadata["class_names"],
        "snr_levels": metadata["snr_levels"],
        "dataset_version": "2018",
    }


def get_available_datasets() -> list[str]:
    """Get list of available datasets."""
    available = []
    if DATA_PATH_2016.exists():
        available.append("RadioML2016.10a")
    # 2018 is available if either preprocessed or raw HDF5 exists
    if is_preprocessed_available() or DATA_PATH_2018.exists():
        available.append("RadioML2018.01a")
    return available


def get_dataset_path(dataset_name: str) -> Path:
    """Get path for a dataset name."""
    if "2018" in dataset_name:
        return DATA_PATH_2018
    return DATA_PATH_2016


def get_dataset_version(dataset_name: str) -> str:
    """Get version string for a dataset name."""
    if "2018" in dataset_name:
        return "2018"
    return "2016"


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
    class_names: list[str] = None,
) -> np.ndarray:
    """Get samples for a specific modulation and SNR.

    Args:
        data: Signal data array or proxy
        labels: Label array
        snrs: SNR array
        modulation: Modulation name to filter by
        snr: SNR value to filter by
        n_samples: Number of samples to return
        normalize: Whether to normalize samples to unit power
        class_names: List of class names (defaults to 2016 classes)

    Returns:
        Array of samples or None if no matching samples found
    """
    if class_names is None:
        class_names = MODULATION_CLASSES

    if modulation not in class_names:
        return None

    mod_idx = class_names.index(modulation)
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
    model: nn.Module,
    signal: np.ndarray,
    normalize: bool = True,
) -> tuple[str, np.ndarray]:
    """Run inference on a signal and return prediction.

    Works with both PFCNN and CLSRAMC models, including 2018 variants.

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

    # Move input to same device as model (handles cached models that may have been
    # moved to MPS/CUDA by other pages)
    device = next(model.parameters()).device
    x = x.to(device)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = logits.argmax(dim=1).item()

    # Determine class names based on model's number of classes
    num_classes = logits.shape[1]
    if num_classes == len(MODULATION_CLASSES_2018):
        class_names = MODULATION_CLASSES_2018
    else:
        class_names = MODULATION_CLASSES

    return class_names[pred_idx], probs
