"""
Train Contrastive TAP-Score Model

Ranking-based objective that fixes the BCE collapse problem.
The model learns to rank the correct action above negatives.

Supports multiple benchmarks: pusht, lift, kitchen, blockpush

Usage:
    python train_contrastive_tap.py --benchmark pusht
    python train_contrastive_tap.py --benchmark lift
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
from tap.benchmarks import get_benchmark_config, get_data_path, list_benchmarks


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
    parser.add_argument("--benchmark", type=str, default="pusht",
                        choices=list_benchmarks(),
                        help="Benchmark to train on")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (default: auto-resolve based on benchmark)")
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
    parser.add_argument("--use_deltas", action="store_true", help="Use delta encoding for state observations")
    parser.add_argument("--use_nn_negatives", action="store_true", help="Use nearest-neighbor hard negatives for state obs")
    parser.add_argument("--nn_negative_ratio", type=float, default=0.3, help="Fraction of negatives from NN states")
    parser.add_argument("--dp_neg_cache", type=str, default=None,
                        help="NPZ cache of DP-proposal negatives (triggers real-negative mode)")
    parser.add_argument("--dp_neg_mixture", action="store_true",
                        help="Mixture mode: expert positives + synthetic + DP negatives")
    parser.add_argument("--dp_neg_ratio", type=float, default=0.3,
                        help="Fraction of negatives from DP cross-obs proposals (mixture mode)")
    parser.add_argument("--learnable_temperature", action="store_true",
                        help="Make InfoNCE temperature a learnable parameter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    # Get benchmark config and auto-resolve data path
    benchmark_config = get_benchmark_config(args.benchmark)
    if args.data_dir is None:
        args.data_dir = str(get_data_path(args.benchmark))

    # Get observation type from benchmark config
    obs_type = benchmark_config.get('obs_type', 'image')

    print("=" * 60)
    print(f"Contrastive TAP-Score Training - {benchmark_config['name']}")
    print("=" * 60)
    print(f"Benchmark:    {args.benchmark}")
    print(f"Obs type:     {obs_type}")
    print(f"Action dim:   {benchmark_config['action_dim']}")
    if obs_type == 'state':
        print(f"Obs dim:      {benchmark_config.get('obs_dim', 'auto')}")
        print(f"Use deltas:   {args.use_deltas}")
        print(f"NN negatives: {args.use_nn_negatives} (ratio: {args.nn_negative_ratio})")
    print("Key: ranking objective (InfoNCE) instead of BCE")
    print("=" * 60)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Create output directory (benchmark-specific)
    output_dir = Path(args.output_dir) / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print(f"\nLoading data from {args.data_dir}...")
    print(f"Negatives per sample: {args.n_negatives} ({int(args.hard_negative_ratio * 100)}% hard)")

    # Build kwargs for dataloader
    dataloader_kwargs = {
        'obs_window': args.obs_window,
        'action_chunk': args.action_chunk,
        'n_negatives': args.n_negatives,
        'hard_negative_ratio': args.hard_negative_ratio,
    }

    # Add state-specific options
    if obs_type == 'state':
        dataloader_kwargs['use_deltas'] = args.use_deltas
        dataloader_kwargs['use_nn_negatives'] = args.use_nn_negatives
        dataloader_kwargs['nn_negative_ratio'] = args.nn_negative_ratio

    # Pass data_format and obs_keys from benchmark config
    data_format = benchmark_config.get('data_format', 'zarr')
    if 'obs_keys' in benchmark_config:
        dataloader_kwargs['obs_keys'] = benchmark_config['obs_keys']

    # Add mixture-mode kwargs
    if args.dp_neg_mixture:
        dataloader_kwargs['dp_neg_ratio'] = args.dp_neg_ratio

    train_loader, val_loader = create_contrastive_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        obs_type=obs_type,
        data_format=data_format,
        dp_neg_cache=args.dp_neg_cache,
        dp_neg_mixture=args.dp_neg_mixture,
        **dataloader_kwargs,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Get action dimension from dataset
    action_dim = train_loader.dataset.action_dim
    print(f"Auto-detected action_dim: {action_dim}")

    # Auto-detect action_chunk from DP-neg dataset if applicable.
    if args.dp_neg_cache is not None and hasattr(train_loader.dataset, 'action_chunk'):
        args.action_chunk = train_loader.dataset.action_chunk
        print(f"Auto-detected action_chunk from cache: {args.action_chunk}")

    # Get obs_dim from dataset for state observations
    obs_dim = None
    if obs_type == 'state' or args.dp_neg_cache is not None:
        obs_dim = train_loader.dataset.obs_dim
        print(f"Auto-detected obs_dim: {obs_dim}")

    # Build model
    print("\nBuilding contrastive model...")
    config = {
        "obs_channels": 3,
        "action_dim": action_dim,
        "obs_window": args.obs_window,
        "action_chunk": args.action_chunk,
        "hidden_dim": args.hidden_dim,
        "temperature": args.temperature,
        "obs_type": obs_type if args.dp_neg_cache is None else 'state',
        "obs_dim": obs_dim,
        "use_deltas": args.use_deltas if obs_type == 'state' else False,
        "learnable_temperature": args.learnable_temperature,
    }
    model = build_contrastive_tap_model(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Temperature: {args.temperature} ({'learnable' if args.learnable_temperature else 'fixed'})")

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

        temp_str = ""
        if args.learnable_temperature:
            temp_str = f" - Temp: {model.get_temperature():.4f}"
        print(f"Epoch {epoch+1}/{args.epochs} - LR: {current_lr:.2e} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}{temp_str}")

        epoch_record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        if args.learnable_temperature:
            epoch_record["temperature"] = model.get_temperature()
        history.append(epoch_record)

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
