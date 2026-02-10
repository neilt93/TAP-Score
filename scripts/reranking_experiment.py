"""
Reranking experiment — the closing loop.

Under occlusion, compare three strategies at each decision point:
  1. K_notap (baseline): execute candidate 0
  2. K_TAP: use trained TAP-Score to pick the best candidate
  3. Oracle best-of-K: pick the candidate with highest actual reward

Report "fraction of oracle headroom captured":
  (TAP - notap) / (oracle - notap)

Usage:
    cd /pennstate-project && PYTHONPATH=/pennstate-project python scripts/reranking_experiment.py \
        --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
        --tap_checkpoint checkpoints_contrastive/pusht/contrastive_tap_best.pt \
        --tap_config checkpoints_contrastive/pusht/config.json \
        --n_episodes 20 --K 8 --L 5 --perturb occlusion --device cuda
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
import torch.nn.functional as F
import dill
import hydra
from tqdm import tqdm

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from tap.env_wrapper import PushTStateWrapper
from tap.contrastive import ContrastiveTAPScore, build_contrastive_tap_model
from tap.perturbations import apply_occlusion, apply_gaussian_noise


# ── helpers ──────────────────────────────────────────────────

def load_diffusion_policy(checkpoint_path: str, device: torch.device):
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
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
    return policy, cfg


def load_tap_model(checkpoint_path: str, config_path: str, device: torch.device):
    with open(config_path) as f:
        config = json.load(f)
    model = build_contrastive_tap_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle training checkpoint format (has 'model_state_dict' key)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, config


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


def perturb_obs_tensor(obs_tensor, perturb_type, prob=0.5):
    if perturb_type is None or "image" not in obs_tensor:
        return obs_tensor
    img = obs_tensor["image"]
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


def compute_progress(block_pose, goal_pose):
    pos_dist = np.linalg.norm(block_pose[:2] - goal_pose[:2])
    angle_diff = (block_pose[2] - goal_pose[2] + math.pi) % (2 * math.pi) - math.pi
    angle_dist = abs(angle_diff)
    pos_score = max(0.0, 1.0 - pos_dist / 362.0)
    angle_score = max(0.0, 1.0 - angle_dist / math.pi)
    return 0.5 * pos_score + 0.5 * angle_score


def _step_branch_chunk(branch_env, branch_obs_hist, chunk, n_action_steps):
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
    branch_env.seed(seed)
    branch_env.reset()
    branch_env.set_state(saved_state)
    return {k: list(v) for k, v in saved_obs_history.items()}


def batched_continuation(branch_envs, branch_obs_histories, branch_dones,
                         branch_max_rewards, dp_policy, n_obs_steps,
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
                _step_branch_chunk, branch_envs[k],
                branch_obs_histories[k], all_chunks[i], n_action_steps
            ))
        for (i, k), fut in zip(enumerate(active), futures):
            max_r, done = fut.result()
            branch_max_rewards[k] = max(branch_max_rewards[k], max_r)
            if done:
                branch_dones[k] = True


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reranking Experiment")
    parser.add_argument("--dp_checkpoint", type=str, required=True)
    parser.add_argument("--tap_checkpoint", type=str, required=True)
    parser.add_argument("--tap_config", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--decision_interval", type=int, default=5)
    parser.add_argument("--skip_first", type=int, default=2)
    parser.add_argument("--max_env_steps", type=int, default=200)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--perturb", type=str, default=None,
                        choices=["occlusion", "noise", "both"])
    parser.add_argument("--perturb_prob", type=float, default=0.5)
    parser.add_argument("--tap_sees_perturb", action="store_true",
                        help="If set, TAP also sees perturbed obs (harder/fairer test). "
                             "Default: TAP sees clean obs.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output is None:
        suffix = f"_perturb_{args.perturb}" if args.perturb else "_clean"
        if args.tap_sees_perturb:
            suffix += "_tap_perturbed"
        args.output = f"eval_results/reranking{suffix}.json"

    device = torch.device(args.device)
    K = args.K
    L = args.L
    perturb_type = args.perturb
    perturb_prob = args.perturb_prob
    tap_sees_perturb = args.tap_sees_perturb

    print("=" * 60)
    print(f"Reranking Experiment (K={K}, L={L})")
    if perturb_type:
        print(f"  Perturbation: {perturb_type} (prob={perturb_prob})")
        print(f"  TAP sees:     {'perturbed' if tap_sees_perturb else 'clean'} obs")
    print("=" * 60)

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load models
    print("\nLoading Diffusion Policy...")
    dp_policy, dp_cfg = load_diffusion_policy(args.dp_checkpoint, device)
    n_action_steps = int(dp_cfg.task.env_runner.n_action_steps)
    n_obs_steps = int(dp_cfg.task.env_runner.n_obs_steps)
    print(f"  n_action_steps={n_action_steps}, n_obs_steps={n_obs_steps}")

    print("Loading TAP-Score model...")
    tap_model, tap_config = load_tap_model(args.tap_checkpoint, args.tap_config, device)
    tap_action_chunk = tap_config["action_chunk"]  # 16
    print(f"  TAP action_chunk={tap_action_chunk}, hidden_dim={tap_config['hidden_dim']}")

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

            # Build clean obs tensor (for TAP and base DP call)
            obs_tensor = {}
            for key in obs.keys():
                stacked = stack_history(obs_history[key], n_obs_steps)
                obs_tensor[key] = torch.from_numpy(stacked[None]).to(device).float()

            if is_decision_point:
                saved_state = env.get_state()
                saved_obs_history = {k: list(v) for k, v in obs_history.items()}

                dp_n_contacts = int(env.n_contact_points)

                # Sample K candidates from DP (with perturbation)
                obs_expanded = {k: v.repeat_interleave(K, dim=0) for k, v in obs_tensor.items()}
                obs_for_dp = {k: v.clone() for k, v in obs_expanded.items()}
                perturb_obs_tensor(obs_for_dp, perturb_type, perturb_prob)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    action_dict = dp_policy.predict_action(obs_for_dp)

                # Get full 16-step predictions for TAP scoring
                action_pred_full = action_dict["action_pred"].detach()  # (K, 16, 2)
                # Get 8-step chunks for env execution
                candidates_exec = action_dict["action"].detach().cpu().numpy()  # (K, 8, 2)

                # ── TAP scoring ──────────────────────────────
                # Default: TAP sees clean obs. With --tap_sees_perturb: same degraded obs as DP.
                if tap_sees_perturb:
                    tap_obs = obs_for_dp["image"][:1]  # (1, T, C, H, W) — perturbed
                else:
                    tap_obs = obs_tensor["image"]  # (1, T, C, H, W) — clean
                # TAP expects actions as (B, M, H, action_dim)
                tap_actions = action_pred_full.unsqueeze(0).float()  # (1, K, 16, 2)

                # Pad or truncate to TAP's expected chunk size
                if tap_actions.shape[2] < tap_action_chunk:
                    pad = torch.zeros(1, K, tap_action_chunk - tap_actions.shape[2], 2,
                                      device=device)
                    tap_actions = torch.cat([tap_actions, pad], dim=2)
                elif tap_actions.shape[2] > tap_action_chunk:
                    tap_actions = tap_actions[:, :, :tap_action_chunk, :]

                tap_logits = tap_model(tap_obs, tap_actions)  # (1, K)
                tap_scores = tap_logits[0].cpu().numpy()  # (K,)
                tap_pick = int(tap_scores.argmax())

                # ── Branch rollouts ──────────────────────────
                init_futures = [
                    executor.submit(_init_branch, env_pool[k], seed, saved_state, saved_obs_history)
                    for k in range(K)
                ]
                branch_obs_histories = [f.result() for f in init_futures]

                step_futures = [
                    executor.submit(
                        _step_branch_chunk, env_pool[k],
                        branch_obs_histories[k], candidates_exec[k], n_action_steps
                    )
                    for k in range(K)
                ]
                branch_dones = []
                branch_max_rewards = []
                for fut in step_futures:
                    max_r, b_done = fut.result()
                    branch_max_rewards.append(max_r)
                    branch_dones.append(b_done)

                # Continuation
                if L > 0:
                    batched_continuation(
                        env_pool[:K], branch_obs_histories, branch_dones,
                        branch_max_rewards, dp_policy, n_obs_steps,
                        n_action_steps, L, device, executor,
                        perturb_type=perturb_type, perturb_prob=perturb_prob,
                    )

                rewards_k = np.array(branch_max_rewards)

                # Record
                all_dps.append({
                    "episode": ep_idx,
                    "seed": seed,
                    "policy_step": policy_step,
                    "env_steps": env_steps,
                    "n_contacts": dp_n_contacts,
                    "rewards_k": rewards_k.tolist(),
                    "tap_scores": tap_scores.tolist(),
                    "tap_pick": tap_pick,
                    "reward_notap": float(rewards_k[0]),
                    "reward_tap": float(rewards_k[tap_pick]),
                    "reward_oracle": float(rewards_k.max()),
                    "reward_spread": float(rewards_k.max() - rewards_k.min()),
                })

                # Continue episode with candidate 0 (baseline trajectory)
                env.set_state(saved_state)
                obs_history = {k: list(v) for k, v in saved_obs_history.items()}
                chunk = candidates_exec[0]
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
    n = len(all_dps)
    r_notap = np.array([d["reward_notap"] for d in all_dps])
    r_tap = np.array([d["reward_tap"] for d in all_dps])
    r_oracle = np.array([d["reward_oracle"] for d in all_dps])

    headroom = r_oracle - r_notap
    tap_gain = r_tap - r_notap

    # Fraction of oracle headroom captured (only where headroom > 0)
    has_headroom = headroom > 0.001
    if has_headroom.sum() > 0:
        capture_ratio = tap_gain[has_headroom] / headroom[has_headroom]
        capture_ratio = np.clip(capture_ratio, 0, 1)  # TAP can't exceed oracle
        mean_capture = float(capture_ratio.mean())
        median_capture = float(np.median(capture_ratio))
    else:
        mean_capture = 0.0
        median_capture = 0.0

    # TAP pick accuracy: how often does TAP pick the oracle best?
    tap_picks_oracle = sum(
        1 for d in all_dps
        if d["reward_tap"] == d["reward_oracle"]
    )

    # TAP beats baseline: how often does TAP pick better than candidate 0?
    tap_beats_baseline = sum(
        1 for d in all_dps
        if d["reward_tap"] > d["reward_notap"] + 0.001
    )

    print("\n" + "=" * 60)
    print("RERANKING RESULTS")
    print("=" * 60)
    print(f"\nDecision points: {n}")
    print(f"With headroom (oracle > notap + 0.001): {has_headroom.sum()}/{n}")

    print(f"\n{'─'*60}")
    print("MEAN REWARDS")
    print(f"{'─'*60}")
    print(f"  K_notap (baseline):  {r_notap.mean():.4f}")
    print(f"  K_TAP (reranked):    {r_tap.mean():.4f}")
    print(f"  Oracle best-of-K:    {r_oracle.mean():.4f}")

    print(f"\n{'─'*60}")
    print("HEADROOM CAPTURE")
    print(f"{'─'*60}")
    print(f"  Mean TAP gain:              {tap_gain.mean():.4f}")
    print(f"  Mean oracle headroom:       {headroom.mean():.4f}")
    print(f"  Mean capture ratio:         {mean_capture:.1%}")
    print(f"  Median capture ratio:       {median_capture:.1%}")
    print(f"  TAP picks oracle best:      {tap_picks_oracle}/{n} ({tap_picks_oracle/n:.1%})")
    print(f"  TAP beats baseline:         {tap_beats_baseline}/{n} ({tap_beats_baseline/n:.1%})")

    # Contact split
    contact_mask = np.array([d["n_contacts"] > 0 for d in all_dps])
    for label, mask in [("Contact", contact_mask), ("No contact", ~contact_mask)]:
        if mask.sum() == 0:
            continue
        hm = headroom[mask]
        tg = tap_gain[mask]
        hh = hm > 0.001
        if hh.sum() > 0:
            cr = np.clip(tg[hh] / hm[hh], 0, 1).mean()
        else:
            cr = 0.0
        print(f"\n  [{label}] (n={mask.sum()})")
        print(f"    Headroom:  {hm.mean():.4f}")
        print(f"    TAP gain:  {tg.mean():.4f}")
        print(f"    Capture:   {cr:.1%}")

    print("\n" + "=" * 60)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "K": K, "L": L, "n_episodes": args.n_episodes,
            "decision_interval": args.decision_interval,
            "skip_first": args.skip_first,
            "perturb": perturb_type, "perturb_prob": perturb_prob,
            "tap_sees_perturb": tap_sees_perturb,
        },
        "summary": {
            "n_decision_points": n,
            "n_with_headroom": int(has_headroom.sum()),
            "mean_reward_notap": float(r_notap.mean()),
            "mean_reward_tap": float(r_tap.mean()),
            "mean_reward_oracle": float(r_oracle.mean()),
            "mean_tap_gain": float(tap_gain.mean()),
            "mean_headroom": float(headroom.mean()),
            "mean_capture_ratio": mean_capture,
            "median_capture_ratio": median_capture,
            "tap_picks_oracle_frac": float(tap_picks_oracle / n),
            "tap_beats_baseline_frac": float(tap_beats_baseline / n),
        },
        "decision_points": all_dps,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
