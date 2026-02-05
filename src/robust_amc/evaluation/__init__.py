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
from .embeddings import (
    compute_cluster_metrics,
    plot_embeddings_tsne,
    plot_embeddings_by_snr,
)
from .sweeps import (
    evaluate_with_impairment,
    sweep_cfo,
    sweep_iq_imbalance,
    sweep_dc_offset,
    sweep_fading,
    plot_impairment_sweep_results,
)
from .cross_dataset import (
    get_class_mapping,
    load_overlapping_data,
    evaluate_cross_dataset,
)

__all__ = [
    # Visualization
    "plot_constellation",
    "plot_accuracy_vs_snr",
    "plot_confusion_matrix",
    "plot_constellation_grid",
    "plot_impairment_sweep",
    "plot_training_history",
    # Metrics
    "evaluate_model",
    "evaluate_snr_sweep",
    "compute_confusion_matrix",
    "accuracy_by_snr",
    "accuracy_by_class",
    "get_embeddings",
    # Embeddings
    "compute_cluster_metrics",
    "plot_embeddings_tsne",
    "plot_embeddings_by_snr",
    # Impairment sweeps
    "evaluate_with_impairment",
    "sweep_cfo",
    "sweep_iq_imbalance",
    "sweep_dc_offset",
    "sweep_fading",
    "plot_impairment_sweep_results",
    # Cross-dataset
    "get_class_mapping",
    "load_overlapping_data",
    "evaluate_cross_dataset",
]
