# Robust Modulation Classification Demonstrator

A demonstrator for robust Automatic Modulation Classification (AMC) using deep learning, based on techniques from the accompanying PhD thesis on domain shift, data augmentation, and contrastive learning.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone and enter directory
cd modulation-classification

# Install dependencies (uv handles virtual environment automatically)
uv sync
```

### Datasets

This project uses two datasets:

**TorchSig (Synthetic)** - Generated automatically on first use
- 5 modulation families: PSK, FSK, AM, SSB, QAM
- Configurable SNR range and impairment levels
- No manual download required

**Panoradio (Real HF)** - Optional, for cross-domain evaluation
```bash
# Download from: https://panoradio-sdr.de/radio-signal-classification-dataset/
# Place files in data/panoradio/:
#   - rscd_2048.npy (5 GB)
#   - tags.csv
```

### Verify Installation

```bash
# Run tests
uv run pytest
```

## Project Structure

```
├── src/robust_amc/          # Main package
│   ├── data/                # Data loading, transforms, augmentations
│   ├── models/              # Neural network architectures
│   ├── losses/              # Loss functions
│   ├── training/            # Training loops and diagnostics
│   ├── evaluation/          # Metrics, visualization, and analysis
│   └── utils/               # Utilities (seeding, device selection)
├── configs/                 # YAML configuration files
│   ├── datasets/            # Dataset configs (torchsig, panoradio)
│   └── label_maps/          # Modulation family mappings
├── scripts/                 # Training and evaluation scripts
├── app/                     # Streamlit demo application
├── checkpoints/             # Saved model weights
├── results/                 # Evaluation outputs
└── tests/                   # Unit tests
```

## Usage

### Training

```bash
# Train PF-CNN on TorchSig with family labels
uv run python scripts/train_pfcnn.py

# Train with MDA-DMC augmentations
uv run python scripts/train_pfcnn.py --augment

# Use custom config
uv run python scripts/train_pfcnn.py --config configs/datasets/torchsig_train.yaml
```

### Cross-Domain Evaluation

```bash
# Evaluate on TorchSig-OOD and Panoradio (zero-shot)
uv run python scripts/evaluate_cross_domain.py --checkpoint checkpoints/pfcnn_torchsig/best_model.pt
```

### Interactive Demo

```bash
uv run streamlit run app/Introduction.py
```

## Modulation Families

Instead of fine-grained modulation labels, this project uses **modulation families** for cross-dataset evaluation:

| Family | TorchSig Examples | Panoradio Examples |
|--------|-------------------|-------------------|
| PSK | BPSK, QPSK, 8PSK | PSK31, QPSK63 |
| FSK | 2FSK, GFSK, MSK | RTTY, Olivia, DominoEx |
| AM | AM-DSB, AM-DSB-SC | AM broadcast |
| SSB | AM-USB, AM-LSB | USB, LSB |
| QAM/OTHER | 16QAM, 64QAM | CW, MT63, Navtex |

## Key Features

- **Family-based classification**: Enables fair cross-dataset evaluation
- **TorchSig integration**: Configurable synthetic signal generation with impairments
- **Panoradio support**: Real HF radio captures for domain transfer evaluation
- **OOD evaluation**: Separate in-distribution and out-of-distribution test sets
- **Config-driven**: YAML-based dataset and experiment configuration

## Development

```bash
# Run tests with coverage
uv run pytest --cov=robust_amc

# Format code
uv run black src tests

# Lint
uv run ruff check src tests
```

## Optional: TorchSig Library

For full TorchSig functionality (advanced signal generation), install from GitHub:

```bash
pip install git+https://github.com/TorchDSP/torchsig.git
```

Without TorchSig, fallback signal generation is used (sufficient for demonstration).
