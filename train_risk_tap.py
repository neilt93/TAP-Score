"""
Train risk-based TAP-Score: P(fail_within_H | obs, action).

Usage:
    python train_risk_tap.py \
        --rollout_dir data/risk_rollouts/can/ \
        --task can --epochs 30 --batch_size 64 \
        --fail_horizon 64 --hard_mining_epoch 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from tap.risk import (
    RiskTAPScore,
    MultiRegimeRiskDataset,
    mine_hard_negatives,
    build_weighted_sampler,
)
from tap.benchmarks import BENCHMARKS


def parse_args():
    parser = argparse.ArgumentParser(description="Train risk-based TAP-Score")
    parser.add_argument("--rollout_dir", type=str, required=True,
                        help="Directory with per-episode NPZ rollouts")
    parser.add_argument("--task", type=str, required=True, choices=["can", "lift", "square"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fail_horizon", type=int, default=64)
    parser.add_argument("--hard_mining_epoch", type=int, default=10,
                        help="Epoch at which to run hard negative mining")
    parser.add_argument("--obs_window", type=int, default=2)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--regimes", nargs="+",
                        default=["clean", "zero80", "freeze80", "dropout80"])
    parser.add_argument("--label_mode", type=str, default="onset_window",
                        choices=["onset_window", "episode_window"],
                        help="Label mode: onset_window (positives only in [onset,onset+H]) "
                             "or episode_window (episode outcome for all steps in window)")
    parser.add_argument("--post_onset_only", action="store_true",
                        help="Only train on post-onset steps (skip pre-onset)")
    parser.add_argument("--balanced", action="store_true",
                        help="Use class-balanced sampling (50/50 pos/neg)")
    parser.add_argument("--obs_only", action="store_true",
                        help="Train obs-only baseline (no action encoder)")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Determine obs_dim and action_dim from benchmark config
    benchmark_name = f"robomimic_{args.task.lower()}_lowdim"
    bench_cfg = BENCHMARKS[benchmark_name]
    obs_dim = bench_cfg["obs_dim"]
    action_dim = bench_cfg["action_dim"]

    print("=" * 70)
    print("Risk TAP-Score Training")
    print("=" * 70)
    print(f"Task:           {args.task} (obs_dim={obs_dim}, action_dim={action_dim})")
    print(f"Rollout dir:    {args.rollout_dir}")
    print(f"Regimes:        {args.regimes}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Fail horizon:   {args.fail_horizon}")
    print(f"Label mode:     {args.label_mode}")
    print(f"Post-onset only:{args.post_onset_only}")
    print(f"Balanced:       {args.balanced}")
    print(f"Hard mining at: epoch {args.hard_mining_epoch}")
    print(f"Obs-only:       {args.obs_only}")
    print(f"Device:         {device}")

    # ── Load dataset ─────────────────────────────────────────────────
    print("\nLoading rollout data...")
    full_dataset = MultiRegimeRiskDataset(
        args.rollout_dir,
        regimes=args.regimes,
        obs_window=args.obs_window,
        action_chunk=args.action_chunk,
        fail_horizon=args.fail_horizon,
        task=args.task,
        label_mode=args.label_mode,
        post_onset_only=args.post_onset_only,
    )

    # ── Split by episode ─────────────────────────────────────────────
    unique_episodes = np.unique(full_dataset.episode_ids)
    np.random.shuffle(unique_episodes)
    n_val = max(1, int(len(unique_episodes) * args.val_split))
    val_episodes = set(unique_episodes[:n_val].tolist())
    train_episodes = set(unique_episodes[n_val:].tolist())

    train_indices = [i for i in range(len(full_dataset))
                     if full_dataset.episode_ids[i] in train_episodes]
    val_indices = [i for i in range(len(full_dataset))
                   if full_dataset.episode_ids[i] in val_episodes]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    n_train_fail = sum(1 for i in train_indices if full_dataset.all_samples[i]["label"] > 0.5)
    n_safe_train = len(train_indices) - n_train_fail
    n_val_fail = sum(1 for i in val_indices if full_dataset.all_samples[i]["label"] > 0.5)

    print(f"\nSplit: {len(train_indices)} train ({n_train_fail} fail, {n_safe_train} safe), "
          f"{len(val_indices)} val ({n_val_fail} fail)")
    print(f"  Episodes: {len(train_episodes)} train, {len(val_episodes)} val")

    # Build initial train loader (optionally balanced)
    if args.balanced:
        train_weights = np.ones(len(train_dataset), dtype=np.float64)
        for i, idx in enumerate(train_indices):
            if full_dataset.all_samples[idx]["label"] > 0.5:
                # Weight positives so they contribute 50% of samples
                train_weights[i] = max(1.0, n_safe_train / max(n_train_fail, 1))
        train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)
        print(f"  Balanced sampling: positive weight = {train_weights.max():.1f}x")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Build model ──────────────────────────────────────────────────
    model = RiskTAPScore(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_window=args.obs_window,
        action_chunk=args.action_chunk,
        hidden_dim=args.hidden_dim,
        obs_only=args.obs_only,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    # ── Checkpoint path ──────────────────────────────────────────────
    ckpt_dir = Path(f"checkpoints_contrastive/{benchmark_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_suffix = "_obs_only" if args.obs_only else ""
    best_path = ckpt_dir / f"risk_tap{ckpt_suffix}_best.pt"

    config = {
        "model_type": "risk",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "obs_window": args.obs_window,
        "action_chunk": args.action_chunk,
        "fail_horizon": args.fail_horizon,
        "hidden_dim": args.hidden_dim,
        "task": args.task,
        "regimes": args.regimes,
        "label_mode": args.label_mode,
        "obs_only": args.obs_only,
    }

    best_val_auroc = 0.0
    t0 = time.time()

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # Hard mining at specified epoch
        if epoch == args.hard_mining_epoch + 1 and args.hard_mining_epoch > 0:
            print(f"\n--- Hard negative mining (epoch {epoch}) ---")
            hard_indices_in_full = mine_hard_negatives(
                model, full_dataset, device, top_k_ratio=0.3,
            )
            # Map to train subset indices (O(1) lookup)
            train_idx_to_pos = {idx: pos for pos, idx in enumerate(train_indices)}
            hard_in_train = [
                train_idx_to_pos[idx] for idx in hard_indices_in_full
                if idx in train_idx_to_pos
            ]
            if len(hard_in_train) > 0:
                # Build weighted sampler preserving balanced weights if active
                if args.balanced:
                    weights = np.ones(len(train_dataset), dtype=np.float64)
                    for i, idx in enumerate(train_indices):
                        if full_dataset.all_samples[idx]["label"] > 0.5:
                            weights[i] = max(1.0, n_safe_train / max(n_train_fail, 1))
                else:
                    weights = np.ones(len(train_dataset), dtype=np.float64)
                for hi in hard_in_train:
                    weights[hi] *= 3.0
                sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size,
                    sampler=sampler, num_workers=0,
                )
                print(f"  Weighted sampler: {len(hard_in_train)} hard samples (3x weight)")
            else:
                print("  No hard samples in train set, continuing with uniform")

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            label = batch["label"].to(device)

            logit = model(obs, action)
            loss = criterion(logit, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(label)
            pred = (logit > 0).float()
            train_correct += (pred == label).sum().item()
            train_total += len(label)

        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_scores = []

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                action = batch["action"].to(device)
                label = batch["label"].to(device)

                logit = model(obs, action)
                loss = criterion(logit, label)

                val_loss += loss.item() * len(label)
                pred = (logit > 0).float()
                val_correct += (pred == label).sum().item()
                val_total += len(label)

                val_labels.extend(label.cpu().numpy().tolist())
                val_scores.extend(logit.cpu().numpy().tolist())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        # AUROC on val
        val_auroc = 0.0
        if len(set(int(l) for l in val_labels)) >= 2:
            val_auroc = roc_auc_score(val_labels, val_scores)

        # Save best
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_auroc": val_auroc,
            }, best_path)
            marker = " *best*"
        else:
            marker = ""

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_auroc={val_auroc:.3f}  "
              f"lr={lr:.6f}{marker}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Best val AUROC: {best_val_auroc:.3f}")
    print(f"Model saved to: {best_path}")


if __name__ == "__main__":
    main()
