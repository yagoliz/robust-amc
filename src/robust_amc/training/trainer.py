"""Training loop and utilities for modulation classification."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from robust_amc.utils import get_device


@dataclass
class TrainingConfig:
    """Configuration for training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    checkpoint_dir: Optional[Path] = None
    device: str = "auto"
    seed: Optional[int] = None

    def __post_init__(self):
        self.device = get_device(self.device)


@dataclass
class TrainingHistory:
    """Training history tracker with diagnostic metrics."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    # Diagnostic fields
    learning_rates: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)


class Trainer:
    """Trainer for modulation classification models.

    Args:
        model: PyTorch model to train
        config: Training configuration
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
        )

        self.history = TrainingHistory()
        self._early_stop_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average loss, accuracy, average gradient norm)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_norms = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            x, y, _ = batch  # Ignore SNR during training
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            # Compute gradient norm before optimizer step
            grad_norm = self._compute_gradient_norm()
            gradient_norms.append(grad_norm)

            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        avg_loss = total_loss / total
        accuracy = correct / total
        avg_grad_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
        return avg_loss, accuracy, avg_grad_norm

    def _compute_gradient_norm(self) -> float:
        """Compute total gradient L2 norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> tuple[float, float]:
        """Evaluate model on a dataset.

        Args:
            data_loader: Data loader to evaluate on

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in data_loader:
            x, y, _ = batch
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
        wandb_logger=None,
    ) -> TrainingHistory:
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Whether to print progress
            wandb_logger: Optional WandbLogger instance for experiment tracking

        Returns:
            Training history
        """
        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_acc, grad_norm = self.train_epoch(train_loader)
            self.history.train_loss.append(train_loss)
            self.history.train_acc.append(train_acc)
            self.history.gradient_norms.append(grad_norm)

            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            self.history.val_loss.append(val_loss)
            self.history.val_acc.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Track learning rate
            lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(lr)

            # Logging
            if verbose:
                print(
                    f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"LR: {lr:.2e}"
                )

            # Log to wandb if enabled
            if wandb_logger is not None:
                wandb_logger.log_metrics(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    learning_rate=lr,
                    gradient_norm=grad_norm,
                )

            # Check for best model
            if val_acc > self.history.best_val_acc:
                self.history.best_val_acc = val_acc
                self.history.best_epoch = epoch + 1
                self._early_stop_counter = 0

                # Save best model
                if self.config.checkpoint_dir:
                    self.save_checkpoint(self.config.checkpoint_dir / "best_model.pt")
            else:
                self._early_stop_counter += 1

            # Early stopping
            if self._early_stop_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if verbose:
            print(f"\nBest validation accuracy: {self.history.best_val_acc:.4f} "
                  f"at epoch {self.history.best_epoch}")

        return self.history

    def save_checkpoint(self, path: Path | str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": {
                "train_loss": self.history.train_loss,
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

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        history = checkpoint["history"]
        self.history.train_loss = history["train_loss"]
        self.history.train_acc = history["train_acc"]
        self.history.val_loss = history["val_loss"]
        self.history.val_acc = history["val_acc"]
        self.history.best_val_acc = history["best_val_acc"]
        self.history.best_epoch = history["best_epoch"]
        # Load new diagnostic fields (with defaults for backward compatibility)
        self.history.learning_rates = history.get("learning_rates", [])
        self.history.gradient_norms = history.get("gradient_norms", [])


def load_model(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: str = "auto",
) -> nn.Module:
    """Load model weights from checkpoint.

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Model with loaded weights
    """
    device = get_device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
