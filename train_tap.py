"""
Train TAP-Score Model

Usage:
    python train_tap.py --data_dir data/processed/pusht
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tap import build_tap_model, TAPDataset, create_dataloaders


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        obs = batch['obs'].to(device)
        actions = batch['actions'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        loss = model.get_loss(obs, actions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        with torch.no_grad():
            scores = model(obs, actions)
            preds = (scores > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            obs = batch['obs'].to(device)
            actions = batch['actions'].to(device)
            labels = batch['label'].to(device)

            loss = model.get_loss(obs, actions, labels)
            total_loss += loss.item()

            scores = model(obs, actions)
            preds = (scores > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Train TAP-Score")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--obs_window", type=int, default=2)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--augment", action="store_true", help="Apply observation augmentations during training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("TAP-Score Training")
    print("=" * 60)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print(f"\nLoading data from {args.data_dir}...")
    print(f"Observation augmentation: {'ENABLED' if args.augment else 'disabled'}")
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        obs_window=args.obs_window,
        action_chunk=args.action_chunk,
        augment=args.augment,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Get action dimension from dataset (auto-detected)
    action_dim = train_loader.dataset.action_dim
    print(f"Auto-detected action_dim: {action_dim}")

    # Build model
    print("\nBuilding model...")
    config = {
        "obs_channels": 3,
        "action_dim": action_dim,  # Auto-detected from data
        "obs_window": args.obs_window,
        "action_chunk": args.action_chunk,
        "hidden_dim": args.hidden_dim,
        "num_layers": 2,
        "num_heads": 4,
    }
    model = build_tap_model(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Learning rate scheduler (cosine annealing for smooth convergence)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01  # Minimum LR is 1% of initial
    )

    # Training loop
    print("\nTraining...")
    best_val_acc = 0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{args.epochs} - LR: {current_lr:.2e} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "val_acc": val_acc,
            }, output_dir / "tap_best.pt")
            print(f"  -> Saved best model (val_acc: {val_acc:.3f})")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "val_acc": val_acc,
    }, output_dir / "tap_final.pt")

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
