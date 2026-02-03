"""Weights & Biases integration for training visualization and experiment tracking."""

from typing import Optional

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger:
    """Logger for Weights & Biases experiment tracking.

    Provides methods to log metrics, embeddings, and visualizations during training.

    Args:
        project: W&B project name
        run_name: Name for this run
        config: Configuration dict to log
        enabled: Whether logging is enabled (allows disabling without code changes)
    """

    def __init__(
        self,
        project: str = "robust-amc",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and WANDB_AVAILABLE

        if self.enabled:
            wandb.init(
                project=project,
                name=run_name,
                config=config or {},
            )
            self._run = wandb.run
        else:
            self._run = None

    def log_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
    ) -> None:
        """Log training metrics for one epoch.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            learning_rate: Current learning rate
            gradient_norm: Optional gradient norm
        """
        if not self.enabled:
            return

        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "train/learning_rate": learning_rate,
            "train/generalization_gap": train_acc - val_acc,
        }

        if gradient_norm is not None:
            metrics["train/gradient_norm"] = gradient_norm

        wandb.log(metrics, step=epoch)

    def log_loss_components(
        self,
        epoch: int,
        contrastive: float,
        reconstruction: float,
        classification: float,
        total: float,
    ) -> None:
        """Log loss components for multi-task learning (CLSR-AMC).

        Args:
            epoch: Current epoch number
            contrastive: Contrastive loss value
            reconstruction: Reconstruction loss value
            classification: Classification loss value
            total: Total weighted loss
        """
        if not self.enabled:
            return

        total_unweighted = contrastive + reconstruction + classification + 1e-8

        wandb.log({
            "loss_components/contrastive": contrastive,
            "loss_components/reconstruction": reconstruction,
            "loss_components/classification": classification,
            "loss_components/total": total,
            "loss_ratios/contrastive": contrastive / total_unweighted,
            "loss_ratios/reconstruction": reconstruction / total_unweighted,
            "loss_ratios/classification": classification / total_unweighted,
        }, step=epoch)

    def log_confusion_matrix(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        class_names: list[str],
        title: str = "Confusion Matrix",
    ) -> None:
        """Log confusion matrix visualization.

        Args:
            targets: True labels
            predictions: Predicted labels
            class_names: List of class names
            title: Title for the visualization
        """
        if not self.enabled:
            return

        wandb.log({
            f"eval/{title.lower().replace(' ', '_')}": wandb.plot.confusion_matrix(
                y_true=targets,
                preds=predictions,
                class_names=class_names,
                title=title,
            )
        })

    def log_snr_accuracy(
        self,
        snr_values: list[int],
        accuracies: list[float],
        model_name: str = "model",
    ) -> None:
        """Log accuracy vs SNR curve.

        Args:
            snr_values: List of SNR values in dB
            accuracies: List of corresponding accuracies
            model_name: Name of the model for the legend
        """
        if not self.enabled:
            return

        # Log as a table for line plot
        table = wandb.Table(
            columns=["SNR (dB)", "Accuracy"],
            data=[[snr, acc] for snr, acc in zip(snr_values, accuracies)]
        )
        wandb.log({
            f"eval/{model_name}_accuracy_vs_snr": wandb.plot.line(
                table, "SNR (dB)", "Accuracy",
                title=f"{model_name} Accuracy vs SNR"
            )
        })

        # Also log individual SNR accuracies
        for snr, acc in zip(snr_values, accuracies):
            wandb.log({f"snr_accuracy/snr_{snr:+d}dB": acc})

    def log_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: list[str],
        snrs: Optional[np.ndarray] = None,
        n_samples: int = 5000,
    ) -> None:
        """Log embeddings for visualization (t-SNE/UMAP in W&B).

        Args:
            embeddings: Embedding array of shape (n_samples, embedding_dim)
            labels: Label array of shape (n_samples,)
            class_names: List of class names
            snrs: Optional SNR values for each sample
            n_samples: Maximum number of samples to log
        """
        if not self.enabled:
            return

        # Subsample if too many points
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
            if snrs is not None:
                snrs = snrs[indices]

        # Create table with embeddings
        columns = [f"emb_{i}" for i in range(embeddings.shape[1])]
        columns.extend(["label", "class_name"])
        if snrs is not None:
            columns.append("snr")

        data = []
        for i in range(len(embeddings)):
            row = list(embeddings[i])
            row.append(int(labels[i]))
            row.append(class_names[labels[i]] if labels[i] < len(class_names) else str(labels[i]))
            if snrs is not None:
                row.append(int(snrs[i]))
            data.append(row)

        table = wandb.Table(columns=columns, data=data)
        wandb.log({"embeddings": table})

    def log_image(self, name: str, image_path: str) -> None:
        """Log an image file.

        Args:
            name: Name for the image in W&B
            image_path: Path to the image file
        """
        if not self.enabled:
            return

        wandb.log({name: wandb.Image(image_path)})

    def log_model_summary(
        self,
        model_name: str,
        n_params: int,
        architecture: Optional[str] = None,
    ) -> None:
        """Log model summary information.

        Args:
            model_name: Name of the model
            n_params: Number of parameters
            architecture: Optional architecture description
        """
        if not self.enabled:
            return

        wandb.config.update({
            "model/name": model_name,
            "model/parameters": n_params,
        })
        if architecture:
            wandb.config.update({"model/architecture": architecture})

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self._run is not None:
            wandb.finish()


def compute_gradient_norm(model) -> float:
    """Compute total gradient L2 norm across all parameters.

    Args:
        model: PyTorch model with gradients computed

    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5