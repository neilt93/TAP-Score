"""
Headroom diagnostic for Path A — oracle best-of-K analysis with:
  1. Contact vs non-contact split
  2. Progress-based label (distance to goal) instead of reward
  3. Oracle vs baseline comparison

Re-runs the same decision points as label_spread_gate but records
block_pose, goal_pose, n_contacts at each decision point and for
each candidate branch.

Usage:
    cd /pennstate-project && PYTHONPATH=/pennstate-project python scripts/headroom_diagnostic.py \
        --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
        --n_episodes 20 --K 8 --L 5 --decision_interval 5 --skip_first 2 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.cuda.amp
import dill
import hydra
from tqdm import tqdm

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from tap.env_wrapper import PushTStateWrapper
from tap.perturbations import apply_occlusion, apply_gaussian_noise


def perturb_obs_tensor(obs_tensor, perturb_type, prob=0.5):
    """Apply perturbation to image obs in-place with probability prob.

    obs_tensor: dict with 'image' key of shape (B, T, C, H, W) on device.
    Perturbation is applied per-sample independently.
    """
    if perturb_type is None or "image" not in obs_tensor:
        return obs_tensor
    img = obs_tensor["image"]  # (B, T, C, H, W)
    img_np = img.cpu().numpy()
    for b in range(img_np.shape[0]):
        if np.random.rand() < prob:
            if perturb_type == "occlusion":
                img_np[b] = apply_occlusion(img_np[b], patch_size=24)
            elif perturb_type == "noise":
                img_np[b] = apply_gaussian_noise(img_np[b], std=0.12)
            elif perturb_type == "both":
                img_np[b] = apply_occlusion(img_np[b], patch_size=24)
                img_np[b] = apply_gaussian_noise(img_np[b], std=0.08)
    obs_tensor["image"] = torch.from_numpy(img_np).to(img.device)
    return obs_tensor


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


def compute_progress(block_pose, goal_pose):
    """Progress = 1 - normalized distance to goal.

    Combines position distance (normalized by workspace diagonal ~724)
    and angle distance (normalized by pi). Weights: 0.5 pos + 0.5 angle.
    Returns value in [0, 1] where 1 = at goal.
    """
    pos_dist = np.linalg.norm(block_pose[:2] - goal_pose[:2])
    # Angle difference wrapped to [-pi, pi]
    angle_diff = (block_pose[2] - goal_pose[2] + math.pi) % (2 * math.pi) - math.pi
    angle_dist = abs(angle_diff)

    # Normalize: workspace is 512x512, diagonal ~724
    pos_score = max(0.0, 1.0 - pos_dist / 362.0)  # half-diagonal
    angle_score = max(0.0, 1.0 - angle_dist / math.pi)

    return 0.5 * pos_score + 0.5 * angle_score


def _step_branch_chunk_with_info(branch_env, branch_obs_hist, chunk, n_action_steps):
    """Step one branch through action chunk, returning reward, progress, and contact info."""
    max_r = 0.0
    max_progress = 0.0
    total_contacts = 0
    done = False
    last_info = None

    for act in chunk[:n_action_steps]:
        o, r, d, info = branch_env.step(act)
        normalize_obs_dict(o)
        for key, val in o.items():
            branch_obs_hist[key].append(val)
        max_r = max(max_r, float(r))
        if info and "block_pose" in info and "goal_pose" in info:
            prog = compute_progress(info["block_pose"], info["goal_pose"])
            max_progress = max(max_progress, prog)
        total_contacts += info.get("n_contacts", 0)
        last_info = info
        if d:
            done = True
            break
    return max_r, max_progress, total_contacts, done, last_info


def _init_branch(branch_env, seed, saved_state, saved_obs_history):
    branch_env.seed(seed)
    branch_env.reset()
    branch_env.set_state(saved_state)
    return {k: list(v) for k, v in saved_obs_history.items()}


def batched_continuation_with_info(branch_envs, branch_obs_histories, branch_dones,
                                   branch_max_rewards, branch_max_progress,
                                   branch_total_contacts, dp_policy, n_obs_steps,
                                   n_action_steps, L, device, executor,
                                   perturb_type=None, perturb_prob=0.5):
    K = len(branch_envs)
    for _ in range(L):
        active = [k for k in range(K) if not branch_dones[k]]
        if not active:
            break

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
        perturb_obs_tensor(batch_tensor, perturb_type, perturb_prob)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            action_dict = dp_policy.predict_action(batch_tensor)
        all_chunks = action_dict["action"].detach().cpu().numpy()

        futures = []
        for i, k in enumerate(active):
            futures.append(executor.submit(
                _step_branch_chunk_with_info, branch_envs[k],
                branch_obs_histories[k], all_chunks[i], n_action_steps
            ))
        for (i, k), fut in zip(enumerate(active), futures):
            max_r, max_prog, contacts, done, _ = fut.result()
            branch_max_rewards[k] = max(branch_max_rewards[k], max_r)
            branch_max_progress[k] = max(branch_max_progress[k], max_prog)
            branch_total_contacts[k] += contacts
            if done:
                branch_dones[k] = True


def main():
    parser = argparse.ArgumentParser(description="Headroom Diagnostic")
    parser.add_argument("--dp_checkpoint", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--decision_interval", type=int, default=5)
    parser.add_argument("--skip_first", type=int, default=2)
    parser.add_argument("--max_env_steps", type=int, default=200)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--perturb", type=str, default=None,
                        choices=["occlusion", "noise", "both"],
                        help="Perturb obs fed to DP (env stays clean)")
    parser.add_argument("--perturb_prob", type=float, default=0.5,
                        help="Probability of applying perturbation per sample")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output is None:
        suffix = f"_perturb_{args.perturb}" if args.perturb else ""
        args.output = f"eval_results/headroom_diagnostic{suffix}.json"

    device = torch.device(args.device)
    K = args.K
    L = args.L
    perturb_type = args.perturb
    perturb_prob = args.perturb_prob

    print("=" * 60)
    print(f"Headroom Diagnostic (K={K}, L={L})")
    if perturb_type:
        print(f"  Perturbation: {perturb_type} (prob={perturb_prob})")
    print("=" * 60)

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("\nLoading Diffusion Policy...")
    dp_policy, dp_cfg = load_diffusion_policy(args.dp_checkpoint, device)
    n_action_steps = int(dp_cfg.task.env_runner.n_action_steps)
    n_obs_steps = int(dp_cfg.task.env_runner.n_obs_steps)
    print(f"  n_action_steps={n_action_steps}, n_obs_steps={n_obs_steps}")

    env_pool = [PushTStateWrapper(PushTImageEnv(legacy=False, render_size=96)) for _ in range(K)]
    executor = ThreadPoolExecutor(max_workers=K)

    seeds = list(range(args.seed_offset, args.seed_offset + args.n_episodes))
    all_dps: List[Dict[str, Any]] = []

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

        while not done and env_steps < args.max_env_steps:
            is_decision_point = (
                policy_step >= args.skip_first
                and (policy_step - args.skip_first) % args.decision_interval == 0
            )

            obs_tensor = {}
            for key in obs.keys():
                stacked = stack_history(obs_history[key], n_obs_steps)
                obs_tensor[key] = torch.from_numpy(stacked[None]).to(device).float()

            if is_decision_point:
                saved_state = env.get_state()
                saved_obs_history = {k: list(v) for k, v in obs_history.items()}

                # Record contact state at decision point
                dp_n_contacts = int(env.n_contact_points)

                # Record block pose and goal pose at decision point
                dp_block_pose = np.array([*env.block.position, env.block.angle])
                dp_goal_pose = np.array(env.goal_pose) if hasattr(env, 'goal_pose') else None
                if dp_goal_pose is None:
                    dp_goal_pose = np.array(env.env.goal_pose)
                dp_progress_at_decision = compute_progress(dp_block_pose, dp_goal_pose)

                # Sample K candidates (perturb obs fed to DP, env stays clean)
                obs_expanded = {k: v.repeat_interleave(K, dim=0) for k, v in obs_tensor.items()}
                perturb_obs_tensor(obs_expanded, perturb_type, perturb_prob)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    action_dict = dp_policy.predict_action(obs_expanded)
                candidates = action_dict["action"].detach().cpu().numpy()

                # Init branches in parallel
                init_futures = [
                    executor.submit(_init_branch, env_pool[k], seed, saved_state, saved_obs_history)
                    for k in range(K)
                ]
                branch_obs_histories = [f.result() for f in init_futures]

                # Execute candidate chunks in parallel
                step_futures = [
                    executor.submit(
                        _step_branch_chunk_with_info, env_pool[k],
                        branch_obs_histories[k], candidates[k], n_action_steps
                    )
                    for k in range(K)
                ]
                branch_dones = []
                branch_max_rewards = []
                branch_max_progress = []
                branch_total_contacts = []
                for fut in step_futures:
                    max_r, max_prog, contacts, b_done, _ = fut.result()
                    branch_max_rewards.append(max_r)
                    branch_max_progress.append(max_prog)
                    branch_total_contacts.append(contacts)
                    branch_dones.append(b_done)

                # Batched continuation
                if L > 0:
                    batched_continuation_with_info(
                        env_pool[:K], branch_obs_histories, branch_dones,
                        branch_max_rewards, branch_max_progress,
                        branch_total_contacts, dp_policy, n_obs_steps,
                        n_action_steps, L, device, executor,
                        perturb_type=perturb_type, perturb_prob=perturb_prob,
                    )

                rewards_k = np.array(branch_max_rewards)
                progress_k = np.array(branch_max_progress)

                all_dps.append({
                    "episode": ep_idx,
                    "seed": seed,
                    "policy_step": policy_step,
                    "env_steps": env_steps,
                    "n_contacts_at_decision": dp_n_contacts,
                    "progress_at_decision": float(dp_progress_at_decision),
                    "rewards_k": rewards_k.tolist(),
                    "progress_k": progress_k.tolist(),
                    "contacts_k": branch_total_contacts,
                    "reward_spread": float(rewards_k.max() - rewards_k.min()),
                    "progress_spread": float(progress_k.max() - progress_k.min()),
                    "reward_baseline": float(rewards_k[0]),
                    "reward_oracle": float(rewards_k.max()),
                    "progress_baseline": float(progress_k[0]),
                    "progress_oracle": float(progress_k.max()),
                })

                # Continue episode with candidate 0
                env.set_state(saved_state)
                obs_history = {k: list(v) for k, v in saved_obs_history.items()}
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
                obs_for_dp = {k: v.clone() for k, v in obs_tensor.items()}
                perturb_obs_tensor(obs_for_dp, perturb_type, perturb_prob)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    action_dict = dp_policy.predict_action(obs_for_dp)
                chunk = action_dict["action"][0].detach().cpu().numpy()
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

    executor.shutdown(wait=False)

    # ========== ANALYSIS ==========
    dps = all_dps
    n = len(dps)

    reward_improvements = np.array([d["reward_oracle"] - d["reward_baseline"] for d in dps])
    progress_improvements = np.array([d["progress_oracle"] - d["progress_baseline"] for d in dps])
    reward_spreads = np.array([d["reward_spread"] for d in dps])
    progress_spreads = np.array([d["progress_spread"] for d in dps])

    # Split by contact
    contact_mask = np.array([d["n_contacts_at_decision"] > 0 for d in dps])
    n_contact = int(contact_mask.sum())
    n_no_contact = n - n_contact

    print("\n" + "=" * 60)
    print("HEADROOM DIAGNOSTIC")
    print("=" * 60)

    print(f"\nTotal decision points: {n}")
    print(f"  With contact:    {n_contact} ({n_contact/n:.0%})")
    print(f"  Without contact: {n_no_contact} ({n_no_contact/n:.0%})")

    # --- Reward-based ---
    print(f"\n{'─'*60}")
    print("REWARD-BASED ORACLE HEADROOM")
    print(f"{'─'*60}")
    for label, mask in [("All", np.ones(n, dtype=bool)),
                        ("Contact", contact_mask),
                        ("No contact", ~contact_mask)]:
        if mask.sum() == 0:
            continue
        ri = reward_improvements[mask]
        rs = reward_spreads[mask]
        print(f"\n  [{label}] (n={mask.sum()})")
        print(f"    Mean oracle improvement:  {ri.mean():.4f}")
        print(f"    Median oracle improvement:{np.median(ri):.4f}")
        print(f"    Max oracle improvement:   {ri.max():.4f}")
        print(f"    Mean spread:              {rs.mean():.4f}")
        print(f"    Frac spread > 0.01:       {float(np.mean(rs > 0.01)):.1%}")
        print(f"    Frac improvement > 0.01:  {float(np.mean(ri > 0.01)):.1%}")

    # --- Progress-based ---
    print(f"\n{'─'*60}")
    print("PROGRESS-BASED ORACLE HEADROOM")
    print(f"{'─'*60}")
    for label, mask in [("All", np.ones(n, dtype=bool)),
                        ("Contact", contact_mask),
                        ("No contact", ~contact_mask)]:
        if mask.sum() == 0:
            continue
        pi = progress_improvements[mask]
        ps = progress_spreads[mask]
        print(f"\n  [{label}] (n={mask.sum()})")
        print(f"    Mean oracle improvement:  {pi.mean():.4f}")
        print(f"    Median oracle improvement:{np.median(pi):.4f}")
        print(f"    Max oracle improvement:   {pi.max():.4f}")
        print(f"    Mean spread:              {ps.mean():.4f}")
        print(f"    Frac spread > 0.01:       {float(np.mean(ps > 0.01)):.1%}")
        print(f"    Frac improvement > 0.01:  {float(np.mean(pi > 0.01)):.1%}")

    # --- Progress at decision point vs headroom ---
    print(f"\n{'─'*60}")
    print("HEADROOM BY EPISODE PHASE")
    print(f"{'─'*60}")
    prog_at = np.array([d["progress_at_decision"] for d in dps])
    for lo, hi, label in [(0.0, 0.3, "Early (progress < 0.3)"),
                          (0.3, 0.6, "Mid (0.3-0.6)"),
                          (0.6, 1.01, "Late (progress > 0.6)")]:
        phase_mask = (prog_at >= lo) & (prog_at < hi)
        if phase_mask.sum() == 0:
            continue
        ri = reward_improvements[phase_mask]
        pi = progress_improvements[phase_mask]
        print(f"\n  [{label}] (n={phase_mask.sum()})")
        print(f"    Reward:   mean improve={ri.mean():.4f}, frac>0.01={float(np.mean(ri>0.01)):.1%}")
        print(f"    Progress: mean improve={pi.mean():.4f}, frac>0.01={float(np.mean(pi>0.01)):.1%}")

    print("\n" + "=" * 60)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {"K": K, "L": L, "n_episodes": args.n_episodes,
                   "decision_interval": args.decision_interval,
                   "skip_first": args.skip_first,
                   "perturb": perturb_type, "perturb_prob": perturb_prob},
        "summary": {
            "n_decision_points": n,
            "n_contact": n_contact,
            "n_no_contact": n_no_contact,
            "reward_mean_improvement": float(reward_improvements.mean()),
            "reward_median_improvement": float(np.median(reward_improvements)),
            "progress_mean_improvement": float(progress_improvements.mean()),
            "progress_median_improvement": float(np.median(progress_improvements)),
            "reward_frac_improvement_gt_0.01": float(np.mean(reward_improvements > 0.01)),
            "progress_frac_improvement_gt_0.01": float(np.mean(progress_improvements > 0.01)),
        },
        "decision_points": all_dps,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
