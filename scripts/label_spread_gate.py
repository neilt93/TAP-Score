"""
Label Spread Gate — go/no-go signal before building anything big.

At each decision point during a DP rollout:
  1. Save env state and obs history
  2. Sample K candidate action chunks from DP
  3. For each candidate k, restore into a fresh env and:
     a. Execute the 8-step candidate chunk
     b. Continue for L more policy steps with deterministic DP
        (fixed RNG seed per branch so noise doesn't dominate)
     c. Record max reward over the full trajectory
  4. Compute spread = max(reward_k) - min(reward_k)

If a decent fraction of spreads are clearly above 0, the ranking signal
exists and we can train a listwise ranker.

Usage:
    python scripts/label_spread_gate.py \
        --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
        --n_episodes 20 --K 8 --L 3 --decision_interval 5 --skip_first 2 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.cuda.amp
import dill
import hydra
from tqdm import tqdm

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from tap.env_wrapper import PushTStateWrapper


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


def _ensure_float_chw(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
        img = np.transpose(img, (2, 0, 1))
    return img


def normalize_obs_dict(obs_dict):
    if "image" in obs_dict:
        obs_dict["image"] = _ensure_float_chw(obs_dict["image"])
    return obs_dict


def stack_history(hist, T):
    if len(hist) < T:
        padded = [hist[0]] * (T - len(hist)) + list(hist)
    else:
        padded = list(hist[-T:])
    return np.stack(padded, axis=0)


def _step_branch_chunk(branch_env, branch_obs_hist, chunk, n_action_steps):
    """Step one branch through an action chunk. Thread-safe (pymunk releases GIL)."""
    max_r = 0.0
    done = False
    for act in chunk[:n_action_steps]:
        o, r, d, info = branch_env.step(act)
        normalize_obs_dict(o)
        for key, val in o.items():
            branch_obs_hist[key].append(val)
        max_r = max(max_r, float(r))
        if d:
            done = True
            break
    return max_r, done


def _init_branch(branch_env, seed, saved_state, saved_obs_history):
    """Reset a pool env and restore saved state. Thread-safe."""
    branch_env.seed(seed)
    branch_env.reset()
    branch_env.set_state(saved_state)
    return {k: list(v) for k, v in saved_obs_history.items()}


def batched_continuation(branch_envs, branch_obs_histories, branch_dones,
                         branch_max_rewards, dp_policy, n_obs_steps,
                         n_action_steps, L, device, executor, t_env, t_dp):
    """Run L continuation policy steps with batched DP inference across all K branches.

    Instead of K sequential forward passes per step, gathers obs from all
    active branches into one batch for a single forward pass per step.
    Env stepping is parallelized across branches using threads.
    """
    K = len(branch_envs)
    for _ in range(L):
        active = [k for k in range(K) if not branch_dones[k]]
        if not active:
            break

        # Gather obs from all active branches into a single batch
        batch_obs = {}
        for k in active:
            obs_hist = branch_obs_histories[k]
            for key in obs_hist:
                stacked = stack_history(obs_hist[key], n_obs_steps)
                tensor = torch.from_numpy(stacked[None]).float()
                if key not in batch_obs:
                    batch_obs[key] = []
                batch_obs[key].append(tensor)

        batch_tensor = {k: torch.cat(v, dim=0).to(device) for k, v in batch_obs.items()}

        # Single batched forward pass
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            action_dict = dp_policy.predict_action(batch_tensor)
        all_chunks = action_dict["action"].detach().cpu().numpy()
        t_dp[0] += time.perf_counter() - t0

        # Parallel env stepping across active branches
        t0 = time.perf_counter()
        futures = []
        for i, k in enumerate(active):
            futures.append(executor.submit(
                _step_branch_chunk, branch_envs[k],
                branch_obs_histories[k], all_chunks[i], n_action_steps
            ))
        for (i, k), fut in zip(enumerate(active), futures):
            max_r, done = fut.result()
            branch_max_rewards[k] = max(branch_max_rewards[k], max_r)
            if done:
                branch_dones[k] = True
        t_env[0] += time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Label Spread Gate")
    parser.add_argument("--dp_checkpoint", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--K", type=int, default=4, help="Candidates per decision point")
    parser.add_argument("--L", type=int, default=0,
                        help="Continuation horizon: L additional policy steps after candidate chunk")
    parser.add_argument("--decision_interval", type=int, default=5,
                        help="Sample a decision point every N policy steps")
    parser.add_argument("--skip_first", type=int, default=2,
                        help="Skip first N policy steps before sampling decision points")
    parser.add_argument("--max_env_steps", type=int, default=200)
    parser.add_argument("--success_threshold", type=float, default=0.80)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"eval_results/label_spread_gate_K{args.K}_L{args.L}.json"

    device = torch.device(args.device)
    K = args.K
    L = args.L

    print("=" * 60)
    print(f"Label Spread Gate (K={K}, L={L})")
    print("=" * 60)
    print(f"Episodes:           {args.n_episodes}")
    print(f"K candidates:       {K}")
    print(f"Continuation (L):   {L} policy steps")
    print(f"Decision interval:  every {args.decision_interval} policy steps")
    print(f"Skip first:         {args.skip_first} policy steps")
    print(f"Device:             {args.device}")
    print("=" * 60)

    # Fast inference switches
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load DP
    print("\nLoading Diffusion Policy...")
    dp_policy, dp_cfg = load_diffusion_policy(args.dp_checkpoint, device)
    n_action_steps = int(dp_cfg.task.env_runner.n_action_steps)
    n_obs_steps = int(dp_cfg.task.env_runner.n_obs_steps)
    print(f"  n_action_steps={n_action_steps}, n_obs_steps={n_obs_steps}")

    # Pre-create env pool for branching (fresh envs avoid arbiter cache issues)
    env_pool = [PushTStateWrapper(PushTImageEnv(legacy=False, render_size=96)) for _ in range(K)]

    seeds = list(range(args.seed_offset, args.seed_offset + args.n_episodes))
    all_spreads: List[float] = []
    all_decision_points: List[Dict[str, Any]] = []
    episode_summaries: List[Dict[str, Any]] = []

    # Timing accumulators: [seconds] (mutable list so inner functions can update)
    t_dp = [0.0]
    t_env = [0.0]

    executor = ThreadPoolExecutor(max_workers=K)

    for ep_idx, seed in enumerate(tqdm(seeds, desc="Episodes")):
        env = PushTStateWrapper(PushTImageEnv(legacy=False, render_size=96))
        env.seed(seed)
        obs = env.reset()
        normalize_obs_dict(obs)

        obs_history = defaultdict(list)
        for key, val in obs.items():
            obs_history[key].append(val)

        policy_step = 0
        env_steps = 0
        done = False
        ep_spreads = []

        while not done and env_steps < args.max_env_steps:
            is_decision_point = (
                policy_step >= args.skip_first
                and (policy_step - args.skip_first) % args.decision_interval == 0
            )

            # Build obs tensor
            obs_tensor = {}
            for key in obs.keys():
                stacked = stack_history(obs_history[key], n_obs_steps)
                obs_tensor[key] = torch.from_numpy(stacked[None]).to(device).float()

            if is_decision_point:
                # Save state before branching
                saved_state = env.get_state()
                saved_obs_history = {k: list(v) for k, v in obs_history.items()}

                # Sample K candidates
                obs_expanded = {k: v.repeat_interleave(K, dim=0) for k, v in obs_tensor.items()}
                t0 = time.perf_counter()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    action_dict = dp_policy.predict_action(obs_expanded)
                candidates = action_dict["action"].detach().cpu().numpy()  # (K, n_action_steps, Da)
                t_dp[0] += time.perf_counter() - t0

                # Init all branch envs in parallel (reset clears arbiter caches)
                t0 = time.perf_counter()
                init_futures = [
                    executor.submit(_init_branch, env_pool[k], seed, saved_state, saved_obs_history)
                    for k in range(K)
                ]
                branch_obs_histories = [f.result() for f in init_futures]

                # Execute candidate chunks on all branches in parallel
                step_futures = [
                    executor.submit(
                        _step_branch_chunk, env_pool[k],
                        branch_obs_histories[k], candidates[k], n_action_steps
                    )
                    for k in range(K)
                ]
                branch_dones = []
                branch_max_rewards = []
                for fut in step_futures:
                    max_r, b_done = fut.result()
                    branch_max_rewards.append(max_r)
                    branch_dones.append(b_done)
                t_env[0] += time.perf_counter() - t0

                # Batched continuation: L more policy steps, one forward pass per step
                if L > 0:
                    batched_continuation(
                        env_pool[:K], branch_obs_histories, branch_dones,
                        branch_max_rewards, dp_policy, n_obs_steps,
                        n_action_steps, L, device, executor, t_env, t_dp,
                    )

                rewards_k = branch_max_rewards

                rewards_k = np.array(rewards_k)
                spread = float(rewards_k.max() - rewards_k.min())
                ep_spreads.append(spread)
                all_spreads.append(spread)

                all_decision_points.append({
                    "episode": ep_idx,
                    "seed": seed,
                    "policy_step": policy_step,
                    "env_steps": env_steps,
                    "rewards_k": rewards_k.tolist(),
                    "spread": spread,
                    "best_reward": float(rewards_k.max()),
                    "worst_reward": float(rewards_k.min()),
                })

                # Restore state and continue with candidate 0 (default policy)
                env.set_state(saved_state)
                obs_history = {k: list(v) for k, v in saved_obs_history.items()}

                # Execute candidate 0 to advance the episode
                chunk = candidates[0]
                for act in chunk:
                    obs, reward, done, info = env.step(act)
                    normalize_obs_dict(obs)
                    for key, val in obs.items():
                        obs_history[key].append(val)
                    env_steps += 1
                    if done or env_steps >= args.max_env_steps:
                        done = True
                        break
            else:
                # Normal policy step (no branching)
                t0 = time.perf_counter()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    action_dict = dp_policy.predict_action(obs_tensor)
                chunk = action_dict["action"][0].detach().cpu().numpy()
                t_dp[0] += time.perf_counter() - t0

                for act in chunk:
                    obs, reward, done, info = env.step(act)
                    normalize_obs_dict(obs)
                    for key, val in obs.items():
                        obs_history[key].append(val)
                    env_steps += 1
                    if done or env_steps >= args.max_env_steps:
                        done = True
                        break

            policy_step += 1

        episode_summaries.append({
            "seed": seed,
            "n_decision_points": len(ep_spreads),
            "mean_spread": float(np.mean(ep_spreads)) if ep_spreads else 0.0,
            "max_spread": float(np.max(ep_spreads)) if ep_spreads else 0.0,
            "n_spread_above_0.01": int(sum(1 for s in ep_spreads if s > 0.01)),
            "n_spread_above_0.05": int(sum(1 for s in ep_spreads if s > 0.05)),
            "n_spread_above_0.10": int(sum(1 for s in ep_spreads if s > 0.10)),
        })

    executor.shutdown(wait=False)

    # Timing
    total_t = t_dp[0] + t_env[0]
    print(f"\n  Time in DP inference: {t_dp[0]:.1f}s ({t_dp[0]/total_t*100:.0f}%)" if total_t > 0 else "")
    print(f"  Time in env stepping: {t_env[0]:.1f}s ({t_env[0]/total_t*100:.0f}%)" if total_t > 0 else "")

    # Analysis
    spreads = np.array(all_spreads)
    print("\n" + "=" * 60)
    print(f"LABEL SPREAD ANALYSIS (K={K}, L={L})")
    print("=" * 60)
    print(f"Total decision points: {len(spreads)}")
    print(f"Mean spread:           {spreads.mean():.4f}")
    print(f"Median spread:         {np.median(spreads):.4f}")
    print(f"Std spread:            {spreads.std():.4f}")
    print(f"Max spread:            {spreads.max():.4f}")
    print(f"p25 / p75:             {np.percentile(spreads, 25):.4f} / {np.percentile(spreads, 75):.4f}")
    print()

    thresholds = [0.001, 0.01, 0.05, 0.10, 0.20]
    for t in thresholds:
        frac = float(np.mean(spreads > t))
        print(f"  Spread > {t:.3f}:  {frac:.1%}  ({int(np.sum(spreads > t))}/{len(spreads)})")

    print()
    frac_01 = float(np.mean(spreads > 0.01))
    if frac_01 > 0.3:
        verdict = "GO"
        print(f"GO — ranking signal exists ({frac_01:.0%} > 30%). Proceed to ranking training.")
    elif frac_01 > 0.1:
        verdict = "MARGINAL"
        print(f"MARGINAL — some signal ({frac_01:.0%}). Try increasing K or L.")
    else:
        verdict = "NO-GO"
        print(f"NO-GO — spreads near zero ({frac_01:.0%} > 0.01). Need more continuation or denser reward.")
    print("=" * 60)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "n_episodes": args.n_episodes,
            "K": K,
            "L": L,
            "decision_interval": args.decision_interval,
            "skip_first": args.skip_first,
            "n_action_steps": n_action_steps,
            "max_env_steps": args.max_env_steps,
        },
        "summary": {
            "verdict": verdict,
            "n_decision_points": len(spreads),
            "mean_spread": float(spreads.mean()),
            "median_spread": float(np.median(spreads)),
            "std_spread": float(spreads.std()),
            "max_spread": float(spreads.max()),
            "frac_above_0.001": float(np.mean(spreads > 0.001)),
            "frac_above_0.01": float(np.mean(spreads > 0.01)),
            "frac_above_0.05": float(np.mean(spreads > 0.05)),
            "frac_above_0.10": float(np.mean(spreads > 0.10)),
        },
        "episodes": episode_summaries,
        "decision_points": all_decision_points,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
