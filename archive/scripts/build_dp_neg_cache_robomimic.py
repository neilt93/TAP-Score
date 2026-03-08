"""
Build DP Negative Cache for TAP Retraining (Robomimic lowdim)

For each expert (obs, action) sample in HDF5, run Diffusion Policy K times
to get K candidate action chunks.  These become real "policy negatives" for
TAP training — no synthetic data.

NO env needed — reads obs directly from HDF5 and feeds to dp_policy.predict_action().
HDF5 data is already in the old-fork layout that DP expects, so no obs compatibility
fixes are required.

Output NPZ:
  obs_windows:    (N, n_obs_steps, obs_dim)   flat state for TAP training
  expert_actions: (N, H, 7)                   ground truth (axis_angle)
  dp_negs:        (N, K, H, 7)                real DP proposals (axis_angle)

Usage:
    python scripts/build_dp_neg_cache_robomimic.py \
        --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
        --hdf5_path data/robomimic/datasets/can/ph/low_dim.hdf5 \
        --task can --K 8 --output data/dp_neg_cache_can.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import dill
import h5py
import hydra
import numpy as np
import torch
from tqdm import tqdm

# Make bundled diffusion-policy and repo root importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
DP_ROOT = REPO_ROOT / "baselines" / "diffusion_policy"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (str(DP_ROOT), str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusion_policy.model.common.rotation_transformer import RotationTransformer

# Reuse helpers from the headroom audit.
from robomimic_headroom_audit import (
    cfg_get,
    load_diffusion_policy,
    undo_transform_action,
)

# obs keys used by the DP lowdim checkpoints (order matters for concat).
_OBS_KEYS = {
    "can": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    "lift": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
}


def main():
    parser = argparse.ArgumentParser(description="Build DP negative cache for TAP retraining (robomimic)")
    parser.add_argument("--dp_checkpoint", type=str, required=True, help="Path to DP .ckpt")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to low_dim.hdf5")
    parser.add_argument("--task", type=str, required=True, choices=["can", "lift"])
    parser.add_argument("--K", type=int, default=8, help="DP candidates per sample")
    parser.add_argument("--action_chunk", type=int, default=0,
                        help="TAP action chunk size (0 = use DP horizon from checkpoint)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DP inference")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    K = args.K
    obs_keys = _OBS_KEYS[args.task]

    # ── Load DP policy ──────────────────────────────────────────────────
    print("Loading Diffusion Policy...")
    dp_policy, dp_cfg = load_diffusion_policy(args.dp_checkpoint, device)

    n_obs_steps = int(cfg_get(dp_cfg.task.env_runner, "n_obs_steps", 2))
    abs_action = bool(cfg_get(dp_cfg.task, "abs_action", False))

    # Determine action horizon from DP config.
    shape_meta = getattr(dp_cfg.task, "shape_meta", None)
    if shape_meta is not None and hasattr(shape_meta, "action"):
        dp_horizon = int(shape_meta.action.shape[0])
    else:
        dp_horizon = int(cfg_get(dp_cfg.task.env_runner, "n_action_steps", 8))
    H = args.action_chunk if args.action_chunk > 0 else dp_horizon

    # Rotation transformer for action conversion (10D rot6d → 7D axis_angle).
    # Must match DP's convention: forward = axis_angle→rot6d, inverse = rot6d→axis_angle.
    rotation_transformer = None
    if abs_action:
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep="rotation_6d"
        )

    print(f"  n_obs_steps={n_obs_steps}, dp_horizon={dp_horizon}, H (TAP chunk)={H}")
    print(f"  abs_action={abs_action}, K={K}")

    # ── Load HDF5 expert data ───────────────────────────────────────────
    print(f"Loading HDF5 from {args.hdf5_path}...")
    with h5py.File(args.hdf5_path, "r") as f:
        demos = f["data"]
        demo_keys = sorted(
            (k for k in demos.keys() if k.startswith("demo_")),
            key=lambda k: int(k.split("_")[1]),
        )

        # Read all demos into memory.
        all_obs_per_key: dict[str, list[np.ndarray]] = {k: [] for k in obs_keys}
        all_actions: list[np.ndarray] = []
        episode_ends: list[int] = []
        cumulative = 0

        for dk in demo_keys:
            demo = demos[dk]
            actions = demo["actions"][:].astype(np.float32)
            all_actions.append(actions)
            for key in obs_keys:
                all_obs_per_key[key].append(demo["obs"][key][:].astype(np.float32))
            cumulative += len(actions)
            episode_ends.append(cumulative)

    # Concatenate across demos.
    obs_arrays = {k: np.concatenate(v, axis=0) for k, v in all_obs_per_key.items()}
    actions_all = np.concatenate(all_actions, axis=0)
    episode_ends = np.array(episode_ends)
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    # Also build flat concatenated obs for TAP training.
    obs_flat = np.concatenate([obs_arrays[k] for k in obs_keys], axis=-1)
    obs_dim = obs_flat.shape[-1]
    action_dim_hdf5 = actions_all.shape[-1]

    n_demos = len(episode_ends)
    n_frames = len(actions_all)
    print(f"  {n_demos} demos, {n_frames} frames")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim_hdf5}")

    # ── Build valid sample indices ──────────────────────────────────────
    min_len = n_obs_steps + H
    valid_indices = []
    for ep_idx in range(n_demos):
        s, e = int(episode_starts[ep_idx]), int(episode_ends[ep_idx])
        ep_len = e - s
        if ep_len < min_len:
            continue
        for t in range(n_obs_steps - 1, ep_len - H):
            global_t = s + t
            valid_indices.append((ep_idx, global_t))

    if args.max_samples > 0 and len(valid_indices) > args.max_samples:
        np.random.seed(args.seed)
        np.random.shuffle(valid_indices)
        valid_indices = valid_indices[: args.max_samples]

    N = len(valid_indices)
    print(f"  {N} valid samples (n_obs_steps={n_obs_steps}, H={H})")

    # ── Prepare output arrays ───────────────────────────────────────────
    out_action_dim = 7  # After undo_transform_action (axis_angle)
    dp_negs = np.zeros((N, K, H, out_action_dim), dtype=np.float32)
    expert_actions = np.zeros((N, H, out_action_dim), dtype=np.float32)
    obs_windows = np.zeros((N, n_obs_steps, obs_dim), dtype=np.float32)

    # ── Generate DP proposals in batches ────────────────────────────────
    print(f"\nGenerating {K} DP candidates per sample (batch_size={args.batch_size})...")
    t0 = time.perf_counter()

    for batch_start in tqdm(range(0, N, args.batch_size), desc="DP inference"):
        batch_end = min(batch_start + args.batch_size, N)
        B = batch_end - batch_start

        # Deterministic RNG per batch for reproducibility.
        torch.manual_seed(args.seed + batch_start)

        # Build per-key obs windows for DP.
        batch_obs: dict[str, list[np.ndarray]] = {k: [] for k in obs_keys}
        for i in range(batch_start, batch_end):
            _ep_idx, global_t = valid_indices[i]
            start_t = global_t - n_obs_steps + 1
            for key in obs_keys:
                window = obs_arrays[key][start_t : global_t + 1]  # (T, key_dim)
                batch_obs[key].append(window)

            # Save flat obs window for TAP.
            obs_windows[i] = obs_flat[start_t : global_t + 1]

            # Save expert action (already 7D in HDF5).
            expert_actions[i] = actions_all[global_t : global_t + H]

        # Stack per-key obs, concatenate into flat "obs" for DP lowdim policy,
        # and expand K times for diverse DP sampling.
        per_key_stacked = []
        for key in obs_keys:
            per_key_stacked.append(np.stack(batch_obs[key], axis=0))  # (B, T, key_dim)
        flat_obs = np.concatenate(per_key_stacked, axis=-1)  # (B, T, obs_dim)
        obs_tensor = torch.from_numpy(flat_obs).to(device=device, dtype=torch.float32)
        obs_expanded = obs_tensor.repeat_interleave(K, dim=0)  # (B*K, T, obs_dim)
        obs_dict_torch = {"obs": obs_expanded}

        with torch.no_grad():
            action_dict = dp_policy.predict_action(obs_dict_torch)

        # (B*K, horizon, raw_action_dim) → (B, K, horizon, raw_action_dim)
        raw_actions = action_dict["action"].detach().cpu().numpy()
        raw_action_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(B, K, -1, raw_action_dim)
        raw_horizon = raw_actions.shape[2]

        # Truncate or pad to TAP action_chunk H.
        if raw_horizon >= H:
            dp_chunk = raw_actions[:, :, :H, :]
        else:
            pad_len = H - raw_horizon
            padding = np.tile(raw_actions[:, :, -1:, :], (1, 1, pad_len, 1))
            dp_chunk = np.concatenate([raw_actions, padding], axis=2)

        # Convert 10D (rot6d) → 7D (axis_angle) if abs_action.
        if abs_action and rotation_transformer is not None:
            converted = np.zeros((B, K, H, out_action_dim), dtype=np.float32)
            for b in range(B):
                for k in range(K):
                    converted[b, k] = undo_transform_action(dp_chunk[b, k], rotation_transformer)
            dp_negs[batch_start:batch_end] = converted
        else:
            dp_negs[batch_start:batch_end] = dp_chunk

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({N * K} total DP samples)")

    # ── Sanity check ────────────────────────────────────────────────────
    print("\nSanity check: expert vs DP candidate distances...")
    n_check = min(1000, N)
    dists = []
    for i in range(n_check):
        for k in range(K):
            d = np.linalg.norm(expert_actions[i] - dp_negs[i, k], axis=-1).mean()
            dists.append(d)
    dists = np.array(dists)
    print(f"  Mean L2 distance (expert vs DP): {dists.mean():.4f}")
    print(f"  Median: {np.median(dists):.4f}")
    print(f"  p25: {np.percentile(dists, 25):.4f}, p75: {np.percentile(dists, 75):.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        obs_windows=obs_windows,
        expert_actions=expert_actions,
        dp_negs=dp_negs,
        K=np.array(K),
        action_chunk=np.array(H),
        obs_dim=np.array(obs_dim),
        task=np.array(args.task),
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
    print(f"  obs_windows:    {obs_windows.shape}")
    print(f"  expert_actions: {expert_actions.shape}")
    print(f"  dp_negs:        {dp_negs.shape}")


if __name__ == "__main__":
    main()
