from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from robust_amc.models.clsr_amc import CLSRAMC, CLSRAMCLoss


@dataclass
class CLSRAMCTrainingHistory:
    """Training history for CLSR-AMC with diagnostic metrics."""
    train_loss: list = field(default_factory=list)
    train_contrastive: list = field(default_factory=list)
    train_reconstruction: list = field(default_factory=list)
    train_classification: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    # Diagnostic fields
    learning_rates: list = field(default_factory=list)
    gradient_norms: list = field(default_factory=list)


class CLSRAMCTrainer:
    """Trainer for CLSR-AMC model."""

    def __init__(
        self,
        model: CLSRAMC,
        criterion: CLSRAMCLoss,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: Path | None = None,
        early_stopping_patience: int = 15,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.history = CLSRAMCTrainingHistory()
        self.early_stopping_patience = early_stopping_patience
        self._early_stop_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_con = 0.0
        total_rec = 0.0
        total_cls = 0.0
        correct = 0
        total = 0
        gradient_norms = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            x_i, x_j, x_orig, y, _ = batch
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)
            x_orig = x_orig.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass for contrastive learning
            z_i, z_j, p_i, p_j = self.model.forward_contrastive(x_i, x_j)

            # Reconstruction from first view's embedding
            x_recon = self.model.decode(z_i)

            # Classification from first view
            logits = self.model.classify(z_i)

            # Compute losses
            losses = self.criterion(p_i, p_j, x_recon, x_orig, logits, y)

            # Backward pass
            losses["total"].backward()

            # Compute gradient norm before optimizer step
            grad_norm = self._compute_gradient_norm()
            gradient_norms.append(grad_norm)

            self.optimizer.step()

            # Track metrics
            batch_size = x_i.size(0)
            total_loss += losses["total"].item() * batch_size
            total_con += losses["contrastive"].item() * batch_size
            total_rec += losses["reconstruction"].item() * batch_size
            total_cls += losses["classification"].item() * batch_size

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += batch_size

            pbar.set_postfix(
                loss=losses["total"].item(),
                acc=correct / total,
            )

        avg_grad_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0

        return {
            "loss": total_loss / total,
            "contrastive": total_con / total,
            "reconstruction": total_rec / total,
            "classification": total_cls / total,
            "accuracy": correct / total,
            "gradient_norm": avg_grad_norm,
        }

    def _compute_gradient_norm(self) -> float:
        """Compute total gradient L2 norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict:
        """Evaluate model on validation set."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            x_i, x_j, x_orig, y, _ = batch
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)
            x_orig = x_orig.to(self.device)
            y = y.to(self.device)

            # Forward pass
            z_i, z_j, p_i, p_j = self.model.forward_contrastive(x_i, x_j)
            x_recon = self.model.decode(z_i)
            logits = self.model.classify(z_i)

            losses = self.criterion(p_i, p_j, x_recon, x_orig, logits, y)

            batch_size = x_i.size(0)
            total_loss += losses["total"].item() * batch_size

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += batch_size

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
        wandb_logger=None,
    ) -> CLSRAMCTrainingHistory:
        """Train the model."""
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history.train_loss.append(train_metrics["loss"])
            self.history.train_contrastive.append(train_metrics["contrastive"])
            self.history.train_reconstruction.append(train_metrics["reconstruction"])
            self.history.train_classification.append(train_metrics["classification"])
            self.history.train_acc.append(train_metrics["accuracy"])
            self.history.gradient_norms.append(train_metrics["gradient_norm"])

            # Validate
            val_metrics = self.evaluate(val_loader)
            self.history.val_loss.append(val_metrics["loss"])
            self.history.val_acc.append(val_metrics["accuracy"])

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            # Track learning rate
            lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(lr)

            # Logging
            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Con: {train_metrics['contrastive']:.4f} | "
                    f"Rec: {train_metrics['reconstruction']:.4f} | "
                    f"Cls: {train_metrics['classification']:.4f} | "
                    f"Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"LR: {lr:.2e}"
                )

            # Log to wandb if enabled
            if wandb_logger is not None:
                wandb_logger.log_metrics(
                    epoch=epoch + 1,
                    train_loss=train_metrics["loss"],
                    train_acc=train_metrics["accuracy"],
                    val_loss=val_metrics["loss"],
                    val_acc=val_metrics["accuracy"],
                    learning_rate=lr,
                    gradient_norm=train_metrics["gradient_norm"],
                )
                wandb_logger.log_loss_components(
                    epoch=epoch + 1,
                    contrastive=train_metrics["contrastive"],
                    reconstruction=train_metrics["reconstruction"],
                    classification=train_metrics["classification"],
                    total=train_metrics["loss"],
                )

            # Check for best model
            if val_metrics["accuracy"] > self.history.best_val_acc:
                self.history.best_val_acc = val_metrics["accuracy"]
                self.history.best_epoch = epoch + 1
                self._early_stop_counter = 0

                if self.checkpoint_dir:
                    self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
            else:
                self._early_stop_counter += 1

            # Early stopping
            if self._early_stop_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if verbose:
            print(f"\nBest validation accuracy: {self.history.best_val_acc:.4f} "
                  f"at epoch {self.history.best_epoch}")

        return self.history

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": {
                "train_loss": self.history.train_loss,
                "train_contrastive": self.history.train_contrastive,
                "train_reconstruction": self.history.train_reconstruction,
                "train_classification": self.history.train_classification,
                "train_acc": self.history.train_acc,
                "val_loss": self.history.val_loss,
                "val_acc": self.history.val_acc,
                "best_val_acc": self.history.best_val_acc,
                "best_epoch": self.history.best_epoch,
                "learning_rates": self.history.learning_rates,
                "gradient_norms": self.history.gradient_norms,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer and scheduler if present
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore history
        if "history" in checkpoint:
            h = checkpoint["history"]
            self.history.train_loss = h.get("train_loss", [])
            self.history.train_contrastive = h.get("train_contrastive", [])
            self.history.train_reconstruction = h.get("train_reconstruction", [])
            self.history.train_classification = h.get("train_classification", [])
            self.history.train_acc = h.get("train_acc", [])
            self.history.val_loss = h.get("val_loss", [])
            self.history.val_acc = h.get("val_acc", [])
            self.history.best_val_acc = h.get("best_val_acc", 0.0)
            self.history.best_epoch = h.get("best_epoch", 0)
            self.history.learning_rates = h.get("learning_rates", [])
            self.history.gradient_norms = h.get("gradient_norms", [])
