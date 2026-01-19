"""
Train Contrastive TAP-Score Model

Ranking-based objective that fixes the BCE collapse problem.
The model learns to rank the correct action above negatives.

Usage:
    python train_contrastive_tap.py --data_dir data/processed/pusht
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from tap.contrastive import (
    build_contrastive_tap_model,
    create_contrastive_dataloaders,
)


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch with InfoNCE loss."""
    model.train()
    total_loss = 0
    total_acc = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        obs = batch['obs'].to(device)
        actions = batch['actions'].to(device)

        optimizer.zero_grad()
        loss, accuracy = model.get_loss(obs, actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy.item()

    return total_loss / len(loader), total_acc / len(loader)


def eval_epoch(model, loader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            obs = batch['obs'].to(device)
            actions = batch['actions'].to(device)

            loss, accuracy = model.get_loss(obs, actions)
            total_loss += loss.item()
            total_acc += accuracy.item()

    return total_loss / len(loader), total_acc / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train Contrastive TAP-Score")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--output_dir", type=str, default="checkpoints_contrastive")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--obs_window", type=int, default=2)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n_negatives", type=int, default=15, help="Number of negative actions per sample")
    parser.add_argument("--hard_negative_ratio", type=float, default=0.5, help="Fraction of hard negatives")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("Contrastive TAP-Score Training")
    print("=" * 60)
    print("Key difference: ranking objective (InfoNCE) instead of BCE")
    print("This prevents collapse to constant output under augmentation")
    print("=" * 60)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print(f"\nLoading data from {args.data_dir}...")
    print(f"Negatives per sample: {args.n_negatives} ({int(args.hard_negative_ratio * 100)}% hard)")
    train_loader, val_loader = create_contrastive_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        obs_window=args.obs_window,
        action_chunk=args.action_chunk,
        n_negatives=args.n_negatives,
        hard_negative_ratio=args.hard_negative_ratio,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Get action dimension from dataset
    action_dim = train_loader.dataset.action_dim
    print(f"Auto-detected action_dim: {action_dim}")

    # Build model
    print("\nBuilding contrastive model...")
    config = {
        "obs_channels": 3,
        "action_dim": action_dim,
        "obs_window": args.obs_window,
        "action_chunk": args.action_chunk,
        "hidden_dim": args.hidden_dim,
        "temperature": args.temperature,
    }
    model = build_contrastive_tap_model(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Temperature: {args.temperature}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Training loop
    print("\nTraining...")
    print("Target: retrieval accuracy should climb steadily (not collapse to 1/16 random)")
    best_val_acc = 0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

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
            }, output_dir / "contrastive_tap_best.pt")
            print(f"  -> Saved best model (val_acc: {val_acc:.3f})")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "val_acc": val_acc,
    }, output_dir / "contrastive_tap_final.pt")

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config for evaluation
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best retrieval accuracy: {best_val_acc:.3f}")
    print(f"  (Random baseline would be {1/(args.n_negatives + 1):.3f})")
    print(f"Models saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
