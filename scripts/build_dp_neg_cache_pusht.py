"""
Build DP Negative Cache for TAP Retraining

For each expert (obs, action) training sample, run Diffusion Policy to generate
K alternative action proposals. These become "policy negatives" — actions that
DP would actually propose but that differ from the expert action.

This teaches TAP to discriminate within DP's candidate distribution, which is
exactly what best-of-K reranking needs.

Output: dp_neg_cache.npz with:
  dp_negs: (N_samples, K, action_chunk, action_dim) float32

Usage:
    python scripts/build_dp_neg_cache_pusht.py \
        --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
        --expert_data data/raw/pusht/pusht_cchi_v7_replay.zarr \
        --K 8 \
        --action_chunk 8 \
        --obs_window 2 \
        --batch_size 64 \
        --output dp_neg_cache.npz \
        --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import dill
import hydra
from tqdm import tqdm


def load_diffusion_policy(checkpoint_path: str, device: torch.device):
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    target = getattr(cfg, "_target_", None)
    if target is None:
        ws_cfg = getattr(cfg, "workspace", None)
        if ws_cfg is not None:
            target = getattr(ws_cfg, "_target_", None)
    if target is None:
        raise ValueError("Could not find workspace _target_ in DP checkpoint config.")

    cls = hydra.utils.get_class(target)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    return policy, cfg


def main():
    parser = argparse.ArgumentParser(description="Build DP negative cache for TAP retraining")
    parser.add_argument("--dp_checkpoint", type=str, required=True)
    parser.add_argument("--expert_data", type=str, required=True)
    parser.add_argument("--K", type=int, default=8, help="DP candidates per sample")
    parser.add_argument("--action_chunk", type=int, default=8, help="TAP action chunk size")
    parser.add_argument("--obs_window", type=int, default=2, help="Observation window size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DP inference")
    parser.add_argument("--output", type=str, default="dp_neg_cache.npz")
    parser.add_argument("--global_seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    K = args.K
    H = args.action_chunk
    T = args.obs_window

    # Load DP
    print("Loading Diffusion Policy...")
    dp_policy, dp_cfg = load_diffusion_policy(args.dp_checkpoint, device)
    n_action_steps = int(dp_cfg.task.env_runner.n_action_steps)
    print(f"  n_action_steps={n_action_steps}")

    # Load expert data
    print("Loading expert data...")
    import zarr
    root = zarr.open_group(args.expert_data, mode="r")
    images = root["data"]["img"][:]       # (N, H, W, C) uint8
    actions = root["data"]["action"][:]   # (N, action_dim)
    episode_ends = root["meta"]["episode_ends"][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    action_dim = actions.shape[-1]
    assert action_dim == 2, f"Expected Push-T action_dim=2, got {action_dim}"
    print(f"  {len(images)} frames, {len(episode_ends)} episodes, action_dim={action_dim}")

    # Build valid indices (same logic as ContrastiveTAPDataset)
    valid_indices = []
    min_len = T + H
    for ep_idx in range(len(episode_ends)):
        s, e = int(episode_starts[ep_idx]), int(episode_ends[ep_idx])
        ep_len = e - s
        if ep_len < min_len:
            continue
        for t in range(T - 1, ep_len - H + 1):
            global_t = s + t
            valid_indices.append((ep_idx, global_t))

    if args.max_samples > 0 and len(valid_indices) > args.max_samples:
        valid_indices = valid_indices[:args.max_samples]

    N = len(valid_indices)
    print(f"  {N} valid samples")

    # Prepare output arrays
    dp_negs = np.zeros((N, K, H, action_dim), dtype=np.float32)
    expert_actions = np.zeros((N, H, action_dim), dtype=np.float32)

    # Process in batches
    print(f"\nGenerating {K} DP candidates per sample (batch_size={args.batch_size})...")
    t0 = time.perf_counter()

    for batch_start in tqdm(range(0, N, args.batch_size), desc="DP inference"):
        batch_end = min(batch_start + args.batch_size, N)
        B = batch_end - batch_start

        # Deterministic RNG per batch for reproducibility
        torch.manual_seed(args.global_seed + batch_start)

        # Build observation tensors for this batch
        batch_images = []
        batch_agent_pos = []

        for i in range(batch_start, batch_end):
            ep_idx, global_t = valid_indices[i]

            # Image obs: (T, C, H, W) float32 [0,1]
            start_t = global_t - T + 1
            img_window = images[start_t:global_t + 1]  # (T, H, W, C)
            img_window = img_window.astype(np.float32) / 255.0
            img_window = np.transpose(img_window, (0, 3, 1, 2))  # (T, C, H, W)
            batch_images.append(img_window)

            # Agent pos: approximate as action[t-1]
            # For obs_window=2 at timestep t: [action[t-2], action[t-1]]
            # Pad with action[episode_start] at beginning
            ep_start = int(episode_starts[ep_idx])
            pos_window = []
            for dt in range(T):
                pos_t = start_t + dt
                # agent_pos at time t ≈ action at time t-1
                action_t = max(ep_start, pos_t - 1)
                pos_window.append(actions[action_t].astype(np.float32))
            batch_agent_pos.append(np.stack(pos_window, axis=0))  # (T, action_dim)

        # Stack into tensors
        img_tensor = torch.from_numpy(np.stack(batch_images, axis=0)).to(device).float()  # (B, T, C, H, W)
        pos_tensor = torch.from_numpy(np.stack(batch_agent_pos, axis=0)).to(device).float()  # (B, T, action_dim)

        # Expand K times for diverse sampling
        img_exp = img_tensor.repeat_interleave(K, dim=0)  # (B*K, T, C, H, W)
        pos_exp = pos_tensor.repeat_interleave(K, dim=0)  # (B*K, T, action_dim)

        obs_dict = {"image": img_exp, "agent_pos": pos_exp}

        with torch.no_grad():
            action_dict = dp_policy.predict_action(obs_dict)

        # (B*K, n_action_steps, action_dim) -> (B, K, n_action_steps, action_dim)
        raw_actions = action_dict["action"].detach().cpu().numpy()
        raw_actions = raw_actions.reshape(B, K, n_action_steps, action_dim)

        # Truncate or pad to TAP action_chunk size
        if n_action_steps >= H:
            dp_chunk = raw_actions[:, :, :H, :]
        else:
            pad_len = H - n_action_steps
            padding = np.tile(raw_actions[:, :, -1:, :], (1, 1, pad_len, 1))
            dp_chunk = np.concatenate([raw_actions, padding], axis=2)

        dp_negs[batch_start:batch_end] = dp_chunk

        # Also save aligned expert action chunks
        for j in range(B):
            _, gt = valid_indices[batch_start + j]
            expert_actions[batch_start + j] = actions[gt:gt + H].astype(np.float32)

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({N * K} total DP samples)")

    # Sanity check: distance between expert actions and DP candidates
    print("\nSanity check: expert vs DP candidate distances...")
    dists = []
    for i in range(min(1000, N)):
        ep_idx, global_t = valid_indices[i]
        expert_chunk = actions[global_t:global_t + H].astype(np.float32)  # (H, action_dim)
        for k in range(K):
            d = np.linalg.norm(expert_chunk - dp_negs[i, k], axis=-1).mean()
            dists.append(d)
    dists = np.array(dists)
    print(f"  Mean L2 distance (expert vs DP): {dists.mean():.4f}")
    print(f"  Median: {np.median(dists):.4f}")
    print(f"  p25: {np.percentile(dists, 25):.4f}, p75: {np.percentile(dists, 75):.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        dp_negs=dp_negs,
        expert_actions=expert_actions,
        valid_indices=np.array(valid_indices),
        K=np.array(K),
        action_chunk=np.array(H),
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
    print(f"  dp_negs shape: {dp_negs.shape}")
    print(f"  expert_actions shape: {expert_actions.shape}")


if __name__ == "__main__":
    main()
