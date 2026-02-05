"""Data loading, transforms, and augmentation modules."""

from .transforms import Normalize, PowerNormalize, Compose

# TorchSig and Panoradio loaders
from .label_mapping import (
    FamilyMapper,
    load_family_mapping,
    get_default_torchsig_mapper,
    get_default_panoradio_mapper,
)
from .torchsig_dataset import (
    TorchSigDataset,
    get_torchsig_loaders,
    generate_torchsig_data,
    load_torchsig_data,
    ImpairmentConfig,
    TRAIN_IMPAIRMENT_CONFIG,
    OOD_IMPAIRMENT_CONFIG,
    DEFAULT_MODULATIONS,
)
from .panoradio_dataset import (
    PanoradioDataset,
    get_panoradio_loaders,
    load_panoradio_data,
    load_panoradio_metadata,
    get_panoradio_by_snr,
    PANORADIO_SAMPLE_RATE,
    PANORADIO_SIGNAL_LENGTH,
    PANORADIO_SNR_LEVELS,
)
from .registry import (
    DatasetType,
    DatasetConfig,
    load_config_from_yaml,
    get_loaders,
    get_loaders_from_yaml,
    DATASET_REGISTRY,
)
from .channels import AWGN, RayleighFading, RicianFading, TimeVaryingRayleigh
from .impairments import (
    CarrierFrequencyOffset,
    IQImbalance,
    DCOffset,
    PhaseNoise,
    SampleRateOffset,
)
from .augmentations import (
    AdditiveGaussianNoise,
    RotationInSignalConstellation,
    StretchingInSignalConstellation,
    RotationAndStretchingInSignalConstellation,
    TimeShift,
    RandomFlip,
    RandomAugmentation,
    MDADMCPipeline,
    AGN,
    RSC,
    SSC,
    RSSC,
)

__all__ = [
    # Label mapping (family abstraction)
    "FamilyMapper",
    "load_family_mapping",
    "get_default_torchsig_mapper",
    "get_default_panoradio_mapper",
    # TorchSig synthetic dataset
    "TorchSigDataset",
    "get_torchsig_loaders",
    "generate_torchsig_data",
    "load_torchsig_data",
    "ImpairmentConfig",
    "TRAIN_IMPAIRMENT_CONFIG",
    "OOD_IMPAIRMENT_CONFIG",
    "DEFAULT_MODULATIONS",
    # Panoradio HF dataset
    "PanoradioDataset",
    "get_panoradio_loaders",
    "load_panoradio_data",
    "load_panoradio_metadata",
    "get_panoradio_by_snr",
    "PANORADIO_SAMPLE_RATE",
    "PANORADIO_SIGNAL_LENGTH",
    "PANORADIO_SNR_LEVELS",
    # Dataset registry
    "DatasetType",
    "DatasetConfig",
    "load_config_from_yaml",
    "get_loaders",
    "get_loaders_from_yaml",
    "DATASET_REGISTRY",
    # Transforms
    "Normalize",
    "PowerNormalize",
    "Compose",
    # Channels
    "AWGN",
    "RayleighFading",
    "RicianFading",
    "TimeVaryingRayleigh",
    # Impairments
    "CarrierFrequencyOffset",
    "IQImbalance",
    "DCOffset",
    "PhaseNoise",
    "SampleRateOffset",
    # Augmentations (MDA-DMC)
    "AdditiveGaussianNoise",
    "RotationInSignalConstellation",
    "StretchingInSignalConstellation",
    "RotationAndStretchingInSignalConstellation",
    "TimeShift",
    "RandomFlip",
    "RandomAugmentation",
    "MDADMCPipeline",
    "AGN",
    "RSC",
    "SSC",
    "RSSC",
]
