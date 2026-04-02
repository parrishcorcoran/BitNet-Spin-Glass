"""Training loop for spin glass replica experiments.

Handles:
- Single replica training with checkpoint saving
- Weight decay turn-off for ternary models
- Cosine annealing LR schedule
- Gradient clipping
- wandb logging (optional)

Usage:
    python -m src.train --size M --weight ternary --activation sign \
        --dataset mnist --seed 42 --wandb
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import (
    WeightMode,
    ActivationMode,
    DatasetName,
    TrainConfig,
    MODEL_SIZES,
    DATASET_INPUT_DIM,
    CHECKPOINTS_DIR,
    checkpoint_epochs,
)
from src.datasets import get_dataloaders
from src.model import SpinGlassMLP


def train_replica(
    size_name: str,
    weight_mode: WeightMode,
    activation_mode: ActivationMode,
    dataset: DatasetName,
    seed: int,
    save_all_epochs: bool = False,
    use_wandb: bool = False,
    lr_override: float | None = None,
) -> dict:
    """Train a single replica and save checkpoints.

    Parameters
    ----------
    size_name : str
        Model size key (e.g. "M").
    weight_mode : WeightMode
        FP or TERNARY.
    activation_mode : ActivationMode
        RELU or SIGN.
    dataset : DatasetName
        Which dataset to train on.
    seed : int
        Random seed for this replica.
    save_all_epochs : bool
        If True, save checkpoint at EVERY epoch (for autocorrelation runs).
    use_wandb : bool
        Whether to log to wandb.
    lr_override : float | None
        Override learning rate (for LR sweep experiments).

    Returns
    -------
    dict
        Final metrics: train_loss, test_loss, test_acc, epochs_trained.
    """
    torch.manual_seed(seed)

    # Config
    cfg = TrainConfig.from_modes(weight_mode, dataset)
    if lr_override is not None:
        cfg.lr = lr_override

    input_dim = DATASET_INPUT_DIM[dataset.value]
    ckpt_epochs = set(range(cfg.epochs + 1)) if save_all_epochs else set(checkpoint_epochs(cfg.epochs))

    # Run ID for file naming
    run_id = f"{size_name}_{weight_mode.value}_{activation_mode.value}_{dataset.value}_seed{seed}"
    ckpt_dir = Path(CHECKPOINTS_DIR) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, test_loader = get_dataloaders(dataset, cfg.batch_size)

    # Model
    model = SpinGlassMLP(size_name, weight_mode, activation_mode, input_dim)
    print(f"[{run_id}] Parameters: {model.count_parameters():,}")

    # Save epoch 0 (random init before ANY training)
    _save_checkpoint(model, ckpt_dir, epoch=0)

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    # wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="bitnet-spin-glass",
            name=run_id,
            config={
                "size": size_name,
                "weight_mode": weight_mode.value,
                "activation_mode": activation_mode.value,
                "dataset": dataset.value,
                "seed": seed,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "params": model.count_parameters(),
            },
        )

    # Training loop
    best_test_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        # Turn off weight decay after specified epoch (ternary)
        if cfg.weight_decay_off_epoch and epoch > cfg.weight_decay_off_epoch:
            for pg in optimizer.param_groups:
                pg["weight_decay"] = 0.0

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_max_norm)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Eval
        test_loss, test_acc = _evaluate(model, test_loader, criterion)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Logging
        if epoch % 10 == 0 or epoch <= 5:
            print(
                f"[{run_id}] Epoch {epoch}/{cfg.epochs} "
                f"| Train Loss {train_loss:.4f} Acc {train_acc:.4f} "
                f"| Test Loss {test_loss:.4f} Acc {test_acc:.4f}"
            )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        # Checkpoint
        if epoch in ckpt_epochs:
            _save_checkpoint(model, ckpt_dir, epoch)

    # Save final metrics
    metrics = {
        "run_id": run_id,
        "size": size_name,
        "weight_mode": weight_mode.value,
        "activation_mode": activation_mode.value,
        "dataset": dataset.value,
        "seed": seed,
        "epochs_trained": cfg.epochs,
        "final_train_loss": train_loss,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "best_test_acc": best_test_acc,
        "params": model.count_parameters(),
    }
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if use_wandb:
        import wandb
        wandb.finish()

    return metrics


def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Compute loss and accuracy on a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_x.size(0)
    return total_loss / total, correct / total


def _save_checkpoint(model: nn.Module, ckpt_dir: Path, epoch: int) -> None:
    """Save model state dict as a checkpoint."""
    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    torch.save(model.state_dict(), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single spin glass replica")
    parser.add_argument("--size", type=str, default="M", choices=list(MODEL_SIZES.keys()))
    parser.add_argument("--weight", type=str, default="fp", choices=["fp", "ternary"])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sign"])
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar10", "random_labels_mnist"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-all-epochs", action="store_true",
                        help="Save checkpoint at every epoch (autocorrelation runs)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--lr-override", type=float, default=None,
                        help="Override learning rate (for LR sweep)")
    args = parser.parse_args()

    train_replica(
        size_name=args.size,
        weight_mode=WeightMode(args.weight),
        activation_mode=ActivationMode(args.activation),
        dataset=DatasetName(args.dataset),
        seed=args.seed,
        save_all_epochs=args.save_all_epochs,
        use_wandb=args.wandb,
        lr_override=args.lr_override,
    )


if __name__ == "__main__":
    main()
