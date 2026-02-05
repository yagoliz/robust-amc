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

Download RadioML2016.10a manually from [DeepSig](https://www.deepsig.ai/datasets/) and place it at `data/RML2016.10a_dict.pkl`.

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
├── scripts/                 # Training scripts
├── notebooks/               # Analysis notebooks
├── app/                     # Streamlit demo application
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
uv run python scripts/train_mda_dmc.py

# Train CLSR-AMC
uv run python scripts/train_clsr_amc.py
```

Use `--seed 42` for reproducible training runs.

### Interactive Demo

```bash
uv run python scripts/run_dashboard.py
# or directly:
uv run streamlit run app/Introduction.py
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

## Analysis Notebooks

After training, explore results using Jupyter notebooks:

```bash
uv run jupyter lab notebooks/
```

Available notebooks:
- `01_data_exploration.ipynb` - Dataset inspection and visualization
- `02_evaluate_impairments.ipynb` - Robustness evaluation under hardware impairments
- `03_compare_experiments.ipynb` - Training run comparison and diagnostics
- `04_analyze_embeddings.ipynb` - t-SNE visualization and cluster analysis
- `05_cross_dataset_evaluation.ipynb` - Cross-dataset domain shift analysis

## W&B Logging

Enable Weights & Biases logging during training:

```bash
uv run python scripts/train_baseline.py --wandb --epochs 50
uv run python scripts/train_mda_dmc.py --wandb --epochs 50
uv run python scripts/train_clsr_amc.py --wandb --epochs 50
```
