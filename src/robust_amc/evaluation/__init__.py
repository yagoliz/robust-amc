"""Evaluation metrics and visualization utilities."""

from .visualization import (
    plot_constellation,
    plot_accuracy_vs_snr,
    plot_confusion_matrix,
    plot_constellation_grid,
    plot_impairment_sweep,
    plot_training_history,
)
from .metrics import (
    evaluate_model,
    evaluate_snr_sweep,
    compute_confusion_matrix,
    accuracy_by_snr,
    accuracy_by_class,
    get_embeddings,
)

__all__ = [
    "plot_constellation",
    "plot_accuracy_vs_snr",
    "plot_confusion_matrix",
    "plot_constellation_grid",
    "plot_impairment_sweep",
    "plot_training_history",
    "evaluate_model",
    "evaluate_snr_sweep",
    "compute_confusion_matrix",
    "accuracy_by_snr",
    "accuracy_by_class",
    "get_embeddings",
]
