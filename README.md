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

### Download Dataset

```bash
uv run python scripts/download_data.py
```

This downloads RadioML2016.10a (~500MB) from DeepSig's public repository.

### Verify Installation

```bash
# Run tests
uv run pytest

# Verify data loading and generate constellation plots
uv run python scripts/verify_data.py
```

## Project Structure

```
├── src/robust_amc/          # Main package
│   ├── data/                # Data loading, transforms, augmentations
│   ├── models/              # Neural network architectures
│   ├── losses/              # Loss functions
│   ├── training/            # Training loops
│   └── evaluation/          # Metrics and visualization
├── scripts/                 # Executable scripts
├── app/                     # Streamlit demo application
├── configs/                 # Configuration files
├── checkpoints/             # Saved model weights
├── results/                 # Evaluation outputs
└── tests/                   # Unit tests
```

## Usage

### Training

```bash
# Train baseline PF-CNN
uv run python scripts/train_baseline.py

# Train with MDA-DMC augmentation
uv run python scripts/train_mda.py

# Train CLSR-AMC
uv run python scripts/train_clsr.py
```

### Interactive Demo

```bash
uv run streamlit run app/app.py
```

## Development

```bash
# Run tests with coverage
uv run pytest --cov=robust_amc

# Format code
uv run black src tests

# Lint
uv run ruff check src tests
```

## Visualizations

```bash
uv run python scripts/train_baseline.py --wandb --epochs 50
uv run python scripts/train_mda_dmc.py --wandb --epochs 50
uv run python scripts/train_clsr_amc.py --wandb --epochs 50

# Compare training runs (after training)
uv run python scripts/compare_experiments.py

# Analyze embeddings (after training)
uv run python scripts/analyze_embeddings.py --model all
```
