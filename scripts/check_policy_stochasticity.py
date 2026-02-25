#!/usr/bin/env python3
"""
Policy stochasticity check for robomimic checkpoints.

Tests whether repeated inference on the same observation produces diverse actions.
This is the #1 validity gate before running the headroom audit:

- If near-zero variance: BC-RNN is effectively deterministic → best-of-K is meaningless
- If non-trivial variance: best-of-K headroom audit is valid

Usage:
    python scripts/check_policy_stochasticity.py \
        --dp_checkpoint path/to/checkpoint.ckpt \
        --K 8 --n_obs 5

Outputs per-observation stats:
    - mean pairwise L2 across K samples
    - per-dimension std
    - whether different seeds produce different actions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
DP_ROOT = REPO_ROOT / "baselines" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

import h5py


def cfg_get(cfg_obj, key, default=None):
    if hasattr(cfg_obj, key):
        return getattr(cfg_obj, key)
    if isinstance(cfg_obj, dict) and key in cfg_obj:
        return cfg_obj[key]
    return default


def resolve_existing_path(path_value: str) -> Path:
    raw = Path(path_value).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([Path.cwd() / raw, REPO_ROOT / raw, DP_ROOT / raw])
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve path '{path_value}'. Tried: {tried}")


def mean_pairwise_l2(samples: np.ndarray) -> float:
    """Mean pairwise L2 over samples, shape: (K, ...)."""
    K = samples.shape[0]
    if K < 2:
        return 0.0
    flat = samples.reshape(K, -1).astype(np.float64)
    diff = flat[:, None, :] - flat[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(K, k=1)
    return float(dist[iu].mean())


def main():
    parser = argparse.ArgumentParser(description="Check policy stochasticity")
    parser.add_argument("--dp_checkpoint", type=str, required=True)
    parser.add_argument("--K", type=int, default=8, help="Number of samples per observation")
    parser.add_argument("--n_obs", type=int, default=5, help="Number of different observations to test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    # Load checkpoint
    print("=" * 60)
    print("Policy Stochasticity Check")
    print("=" * 60)
    print(f"Checkpoint: {args.dp_checkpoint}")

    payload = torch.load(args.dp_checkpoint, map_location=device, pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]

    target = getattr(cfg, "_target_", None)
    if target is None:
        ws_cfg = getattr(cfg, "workspace", None)
        if ws_cfg is not None:
            target = getattr(ws_cfg, "_target_", None)

    cls = hydra.utils.get_class(target)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()

    policy_class = type(policy).__name__
    horizon = int(cfg_get(cfg, "horizon", -1))
    n_obs_steps = int(cfg_get(cfg.task.env_runner, "n_obs_steps", 1))
    n_action_steps = int(cfg_get(cfg.task.env_runner, "n_action_steps", 1))
    task_name = str(cfg_get(cfg.task, "task_name", "unknown"))

    print(f"PolicyClass: {policy_class}")
    print(f"Horizon:     {horizon}")
    print(f"Task:        {task_name}")
    print(f"n_obs_steps: {n_obs_steps}")
    print(f"n_action_steps: {n_action_steps}")
    print(f"K:           {args.K}")
    print(f"n_obs:       {args.n_obs}")
    print(f"Device:      {device}")

    # Check if policy has explicit sampling method
    has_sample = hasattr(policy, "sample") or hasattr(policy, "sample_action")
    print(f"Has sample(): {has_sample}")
    print("=" * 60)

    # Load observations from HDF5 dataset (avoids mujoco_py dependency)
    task_cfg = cfg.task
    dataset_path_cfg = str(cfg_get(task_cfg, "dataset_path"))
    dataset_path = resolve_existing_path(dataset_path_cfg)
    obs_keys = list(task_cfg.obs_keys)

    print(f"Dataset: {dataset_path}")
    print(f"Obs keys: {obs_keys}")

    # Load a few observations from different demos
    hdf5_obs_list = []
    with h5py.File(str(dataset_path), "r") as f:
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        rng = np.random.RandomState(42)
        chosen = rng.choice(len(demo_keys), size=min(args.n_obs, len(demo_keys)), replace=False)
        for idx in chosen:
            dk = demo_keys[idx]
            obs_dict = {}
            for ok in obs_keys:
                obs_dict[ok] = np.asarray(f["data"][dk]["obs"][ok][0], dtype=np.float32)
            hdf5_obs_list.append(obs_dict)
    print(f"Loaded {len(hdf5_obs_list)} observations from {len(demo_keys)} demos")

    all_spreads = []
    all_per_dim_stds = []

    for obs_idx in range(len(hdf5_obs_list)):
        obs_dict = hdf5_obs_list[obs_idx]

        # Concatenate all obs keys into single vector (policy expects {'obs': ...})
        obs_concat = np.concatenate([obs_dict[k] for k in obs_keys], axis=-1)
        stacked = np.stack([obs_concat] * n_obs_steps, axis=0)  # (n_obs_steps, obs_dim)
        obs_tensor = {"obs": torch.from_numpy(stacked[None]).to(device=device, dtype=torch.float32)}

        # Sample K actions from same observation
        # Method 1: Batch query (how the audit does it)
        obs_expanded = {k: v.repeat_interleave(args.K, dim=0) for k, v in obs_tensor.items()}
        with torch.no_grad():
            action_dict = policy.predict_action(obs_expanded)
        batch_samples = action_dict["action"].detach().cpu().numpy()  # (K, horizon, action_dim)

        # Method 2: Sequential with different seeds
        seq_samples = []
        for k in range(args.K):
            torch.manual_seed(k * 12345 + obs_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(k * 12345 + obs_idx)
            with torch.no_grad():
                ad = policy.predict_action(obs_tensor)
            seq_samples.append(ad["action"][0].detach().cpu().numpy())
        seq_samples = np.stack(seq_samples, axis=0)  # (K, horizon, action_dim)

        # Compute diversity metrics
        batch_spread = mean_pairwise_l2(batch_samples)
        seq_spread = mean_pairwise_l2(seq_samples)

        # Per-dimension std across K samples (batch method)
        per_dim_std = batch_samples.std(axis=0).mean(axis=0)  # (action_dim,)

        all_spreads.append((batch_spread, seq_spread))
        all_per_dim_stds.append(per_dim_std)

        print(f"\nObs {obs_idx}:")
        print(f"  Batch L2 spread (K={args.K}): {batch_spread:.6f}")
        print(f"  Seeded L2 spread (K={args.K}): {seq_spread:.6f}")
        print(f"  Per-dim std: {per_dim_std}")
        print(f"  Max per-dim std: {per_dim_std.max():.6f}")

        # Check if batch samples are all identical
        if batch_spread < 1e-8:
            print("  WARNING: Batch samples are IDENTICAL (deterministic inference)")
        if seq_spread < 1e-8:
            print("  WARNING: Seeded samples are IDENTICAL (even with different seeds)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    batch_spreads = [s[0] for s in all_spreads]
    seq_spreads = [s[1] for s in all_spreads]
    mean_per_dim = np.mean(all_per_dim_stds, axis=0)

    print(f"Mean batch L2 spread:  {np.mean(batch_spreads):.6f}")
    print(f"Mean seeded L2 spread: {np.mean(seq_spreads):.6f}")
    print(f"Mean per-dim std:      {mean_per_dim}")
    print(f"Max per-dim std:       {mean_per_dim.max():.6f}")

    # Verdict
    THRESHOLD = 1e-4
    batch_stochastic = np.mean(batch_spreads) > THRESHOLD
    seed_stochastic = np.mean(seq_spreads) > THRESHOLD

    print(f"\nBatch stochastic (spread > {THRESHOLD}): {batch_stochastic}")
    print(f"Seed stochastic  (spread > {THRESHOLD}): {seed_stochastic}")

    if batch_stochastic:
        print("\nVERDICT: Policy produces diverse candidates in batch mode.")
        print("  -> Headroom audit is VALID for best-of-K reranking.")
    elif seed_stochastic:
        print("\nVERDICT: Policy is deterministic in batch but diverse with different seeds.")
        print("  -> Headroom audit should use per-candidate seeding (already does).")
    else:
        print("\nVERDICT: Policy is DETERMINISTIC.")
        print("  -> Best-of-K headroom audit will show ZERO headroom.")
        print("  -> Reranking is NOT meaningful for this checkpoint.")
        print("  -> Consider: (a) use a diffusion-policy checkpoint instead of BC-RNN,")
        print("               (b) add action noise for diversity,")
        print("               (c) report this as a negative result.")

    print("=" * 60)


if __name__ == "__main__":
    main()
