"""Training utilities and loops."""

from .clsr_trainer import CLSRAMCTrainer
from .trainer import Trainer, TrainingConfig, TrainingHistory, load_model
from .wandb_logger import WandbLogger, compute_gradient_norm
from .diagnostics import (
    load_history_from_checkpoint,
    plot_multi_run_comparison,
    plot_loss_component_breakdown,
    plot_comprehensive_diagnostic,
    analyze_training_diagnostics,
)

__all__ = [
    # Trainers
    "CLSRAMCTrainer",
    "Trainer",
    "TrainingConfig",
    "TrainingHistory",
    "load_model",
    # Logging
    "WandbLogger",
    "compute_gradient_norm",
    # Diagnostics
    "load_history_from_checkpoint",
    "plot_multi_run_comparison",
    "plot_loss_component_breakdown",
    "plot_comprehensive_diagnostic",
    "analyze_training_diagnostics",
]
