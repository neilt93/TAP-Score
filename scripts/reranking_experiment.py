"""
Reranking experiment — the closing loop.

Under occlusion, compare three strategies at each decision point:
  1. K_notap (baseline): execute candidate 0
  2. K_TAP: use trained TAP-Score to pick the best candidate
  3. Oracle best-of-K: pick the candidate with highest actual reward

Report "fraction of oracle headroom captured":
  (TAP - notap) / (oracle - notap)

Usage:
    # Tier 0 baseline (vanilla DP; TAP disabled)
    python scripts/reranking_experiment.py \
        --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
        --n_episodes 200 --K 1 --L 5 --seed_offset 0 --perturb_seed 123 \
        --output outputs/baseline_clean_n200.json

    python scripts/reranking_experiment.py \
        --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
        --tap_checkpoint checkpoints_contrastive/pusht/contrastive_tap_best.pt \
        --tap_config checkpoints_contrastive/pusht/config.json \
        --n_episodes 20 --K 8 --L 5 --perturb occlusion --patch_size 24 --device cuda
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
from tap.perturbations import sample_patch_xy


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


def _infer_action_chunk_from_state_dict(state_dict: Dict[str, torch.Tensor], action_dim: int) -> int:
    """Infer action_chunk from first action encoder layer input width."""
    key = "action_encoder.mlp.0.weight"
    if key not in state_dict:
        raise ValueError(f"Missing expected key '{key}' in checkpoint state_dict.")
    in_dim = int(state_dict[key].shape[1])
    if in_dim % action_dim != 0:
        raise ValueError(
            f"Invalid action encoder input dim {in_dim} for action_dim {action_dim}; cannot infer action_chunk."
        )
    return in_dim // action_dim


def load_tap_model(checkpoint_path: str, config_path: str | None, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config") if isinstance(ckpt, dict) else None
    ext_config = None
    if config_path is not None:
        with open(config_path) as f:
            ext_config = json.load(f)

    # Prefer checkpoint-embedded config; only use external config as fallback/consistency check.
    if ckpt_config is not None:
        config = dict(ckpt_config)
        if ext_config is not None:
            keys = [
                "action_chunk", "action_dim", "obs_window", "obs_channels",
                "hidden_dim", "temperature", "obs_type", "obs_dim", "use_deltas",
            ]
            mismatches = []
            for k in keys:
                if k in ext_config and k in ckpt_config and ext_config[k] != ckpt_config[k]:
                    mismatches.append((k, ext_config[k], ckpt_config[k]))
            if mismatches:
                details = ", ".join([f"{k}: ext={v1} ckpt={v2}" for k, v1, v2 in mismatches])
                raise ValueError(
                    f"TAP config mismatch between --tap_config and checkpoint-embedded config ({details}). "
                    "Use checkpoint config or matching config file."
                )
    elif ext_config is not None:
        config = dict(ext_config)
    else:
        raise ValueError("No TAP config found: checkpoint has no embedded config and --tap_config was not provided.")

    model = build_contrastive_tap_model(config)
    # Handle training checkpoint format (has 'model_state_dict' key)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    # Hard consistency check: config action_chunk must match model weights.
    action_dim = int(config.get("action_dim", 2))
    inferred_chunk = _infer_action_chunk_from_state_dict(state_dict, action_dim)
    config_chunk = int(config.get("action_chunk", inferred_chunk))
    if config_chunk != inferred_chunk:
        raise ValueError(
            f"TAP checkpoint/config mismatch: config action_chunk={config_chunk}, "
            f"but weights imply action_chunk={inferred_chunk}."
        )
    config["action_chunk"] = inferred_chunk

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


def decision_perturb_seed(base_seed: int, episode_seed: int, policy_step: int) -> int:
    """Deterministic seed for decision-point observation perturbation."""
    return int(base_seed + episode_seed * 10000 + policy_step)


def continuation_perturb_seed(base_seed: int, episode_seed: int, policy_step: int) -> int:
    """Deterministic seed source for continuation-step perturbations."""
    return int(base_seed + episode_seed * 10000 + policy_step * 100)


def mean_ci95(values: np.ndarray) -> Dict[str, float]:
    """Return mean and 95% normal-approx CI for a 1D array."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": 1}
    se = float(arr.std(ddof=1) / np.sqrt(n))
    half = 1.96 * se
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half, "n": n}


def perturb_single_obs(img_np: np.ndarray, perturb_type: str, rng: np.random.Generator,
                       patch_size: int = 24, prob: float = 0.5,
                       xy: tuple = None) -> np.ndarray:
    """Perturb a single observation image. Returns perturbed copy or original.

    img_np: (T, C, H, W) single observation window.
    xy: optional (x, y) to fix the occlusion patch position (episode mode).
        If provided, skips sampling a new position from rng.
    """
    if rng.random() < prob:
        out = img_np.copy()
        occ_slice = None
        if perturb_type in ("occlusion", "both"):
            _, _, H, W = out.shape
            if xy is not None:
                x, y = xy
            else:
                x, y = sample_patch_xy(H, W, patch_size, patch_size, rng)
            occ_slice = (slice(None), slice(None),
                         slice(y, y + patch_size), slice(x, x + patch_size))
            out[occ_slice] = 0.0
        if perturb_type in ("noise", "both"):
            std = 0.08 if perturb_type == "both" else 0.12
            noise = rng.standard_normal(out.shape, dtype=np.float32) * std
            out = np.clip(out + noise, 0.0, 1.0)
            if occ_slice is not None:
                out[occ_slice] = 0.0
        return out
    return img_np


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
                         perturb_type=None, perturb_prob=0.5,
                         patch_size=24, cont_rng=None, xy=None):
    """Continue L policy steps on each branch. Each branch gets its own observation
    (from its own env), but perturbation is applied with a shared RNG for reproducibility."""
    K = len(branch_envs)
    for l_step in range(L):
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

        # ONE camera → same perturbation fate for all branches at each step.
        # Draw one seed per continuation step; all branches share the coin flip
        # (perturb or not) and patch position.  Images differ (different states).
        if perturb_type is not None and "image" in batch_tensor:
            if cont_rng is None:
                raise ValueError("cont_rng must be provided when perturb_type is set.")
            img = batch_tensor["image"]
            img_np = img.cpu().numpy()
            step_seed = cont_rng.integers(0, 2**31) if cont_rng is not None else None
            for b in range(img_np.shape[0]):
                branch_rng = np.random.default_rng(step_seed)  # same seed → same flip/patch
                img_np[b] = perturb_single_obs(
                    img_np[b], perturb_type, branch_rng,
                    patch_size=patch_size, prob=perturb_prob, xy=xy,
                )
            batch_tensor["image"] = torch.from_numpy(img_np).to(img.device)

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
    parser.add_argument("--tap_checkpoint", type=str, default=None,
                        help="Optional TAP checkpoint. If omitted, runs vanilla DP baseline.")
    parser.add_argument("--tap_config", type=str, default=None,
                        help="Optional TAP config. If omitted, checkpoint-embedded config is used.")
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
    parser.add_argument("--patch_size", type=int, default=24,
                        help="Occlusion patch size in pixels")
    parser.add_argument("--perturb_seed", type=int, default=42,
                        help="Seed for perturbation RNG (reproducible masks)")
    parser.add_argument("--occlusion_mode", type=str, default="step",
                        choices=["step", "episode"],
                        help="step: new random patch each policy step. "
                             "episode: one fixed patch for entire episode (tape on lens).")
    parser.add_argument("--tap_sees_perturb", action="store_true",
                        help="If set, TAP also sees perturbed obs (harder/fairer test). "
                             "Default: TAP sees clean obs.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.tap_config is not None and args.tap_checkpoint is None:
        raise ValueError("--tap_config requires --tap_checkpoint.")
    use_tap = args.tap_checkpoint is not None

    if args.output is None:
        prob_tag = str(args.perturb_prob).replace(".", "p")
        seed_tag = f"seed{args.perturb_seed}"
        if args.perturb is None:
            condition_tag = "clean"
        else:
            condition_tag = f"perturb_{args.perturb}_p{prob_tag}_patch{args.patch_size}"
        if use_tap:
            tap_view_tag = "tap_perturbed" if args.tap_sees_perturb else "tap_clean"
        else:
            tap_view_tag = "tap_disabled"
        args.output = (
            f"eval_results/reranking_{condition_tag}_{tap_view_tag}_"
            f"K{args.K}_L{args.L}_di{args.decision_interval}_skip{args.skip_first}_{seed_tag}.json"
        )

    device = torch.device(args.device)
    K = args.K
    L = args.L
    perturb_type = args.perturb
    perturb_prob = args.perturb_prob
    patch_size = args.patch_size
    tap_sees_perturb = args.tap_sees_perturb
    occlusion_mode = args.occlusion_mode

    # Reproducible perturbation RNG — derive per-episode, per-step seeds from this
    perturb_base_seed = args.perturb_seed

    print("=" * 60)
    print(f"Reranking Experiment (K={K}, L={L})")
    if perturb_type:
        print(f"  Perturbation: {perturb_type} (prob={perturb_prob}, patch_size={patch_size})")
        print(f"  Occlusion mode: {occlusion_mode}")
        print(f"  Perturb seed: {perturb_base_seed}")
    if use_tap:
        print(f"  TAP sees:     {'perturbed' if tap_sees_perturb else 'clean'} obs")
    else:
        print("  TAP mode:     disabled (vanilla DP baseline)")
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

    tap_model = None
    tap_config = None
    tap_action_chunk = None
    if use_tap:
        print("Loading TAP-Score model...")
        tap_model, tap_config = load_tap_model(args.tap_checkpoint, args.tap_config, device)
        tap_action_chunk = tap_config["action_chunk"]  # 16
        print(f"  TAP action_chunk={tap_action_chunk}, hidden_dim={tap_config['hidden_dim']}")
        tap_obs_window = int(tap_config.get("obs_window", n_obs_steps))
        if tap_obs_window != n_obs_steps:
            raise ValueError(
                f"Obs-window mismatch: TAP expects obs_window={tap_obs_window}, DP runtime uses n_obs_steps={n_obs_steps}."
            )

    env_pool = [PushTStateWrapper(PushTImageEnv(legacy=False, render_size=96)) for _ in range(K)]
    executor = ThreadPoolExecutor(max_workers=K)

    seeds = list(range(args.seed_offset, args.seed_offset + args.n_episodes))
    all_dps: List[Dict[str, Any]] = []

    for ep_idx, seed in enumerate(tqdm(seeds, desc="Episodes")):
        env = PushTStateWrapper(PushTImageEnv(legacy=False, render_size=96))
        env.seed(seed)
        obs = env.reset()
        normalize_obs_dict(obs)

        # Episode-mode occlusion: one fixed patch for the entire episode
        episode_xy = None
        if perturb_type in ("occlusion", "both") and occlusion_mode == "episode":
            ep_rng = np.random.default_rng(perturb_base_seed + seed * 10000)
            episode_xy = sample_patch_xy(96, 96, patch_size, patch_size, ep_rng)

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

                # ── Perturb ONCE, share across K candidates ──
                # Generate one perturbed observation, then replicate to K.
                # This models reality: one camera frame, K denoising samples.
                clean_obs_np = obs_tensor["image"][0].cpu().numpy()  # (T, C, H, W)

                if perturb_type is not None:
                    # Deterministic per-seed, per-step RNG for reproducibility
                    step_rng = np.random.default_rng(
                        decision_perturb_seed(perturb_base_seed, seed, policy_step)
                    )
                    perturbed_obs_np = perturb_single_obs(
                        clean_obs_np, perturb_type, step_rng,
                        patch_size=patch_size, prob=perturb_prob,
                        xy=episode_xy,
                    )
                else:
                    perturbed_obs_np = clean_obs_np

                # Replicate the SAME perturbed obs to K copies for DP
                perturbed_obs_t = torch.from_numpy(perturbed_obs_np).to(device).float()
                obs_for_dp = {k: v.repeat_interleave(K, dim=0) for k, v in obs_tensor.items()}
                obs_for_dp["image"] = perturbed_obs_t.unsqueeze(0).expand(K, -1, -1, -1, -1)
                # Invariant: one camera observation per decision point, shared across K candidates.
                shared_obs = obs_for_dp["image"][0:1].expand_as(obs_for_dp["image"])
                if not torch.allclose(obs_for_dp["image"], shared_obs):
                    raise RuntimeError("Shared-perturbation invariant violated: DP candidates saw different observations.")

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    action_dict = dp_policy.predict_action(obs_for_dp)

                # Get 8-step chunks for env execution
                candidates_exec = action_dict["action"].detach().cpu().numpy()  # (K, 8, 2)

                # ── TAP scoring ──────────────────────────────
                if use_tap:
                    # Default: TAP sees clean obs. With --tap_sees_perturb: same perturbed obs.
                    if tap_sees_perturb:
                        tap_obs = obs_for_dp["image"][0:1]  # exact DP-conditioned perturbed obs
                    else:
                        tap_obs = obs_tensor["image"]  # (1, T, C, H, W) — clean
                    # TAP expects actions as (B, M, H, action_dim).
                    # If TAP H matches executed chunk length, score exact executed candidates.
                    if tap_action_chunk == candidates_exec.shape[1]:
                        tap_actions_np = candidates_exec
                    else:
                        # Fallback to full DP prediction horizon when TAP H differs.
                        action_pred_full = action_dict["action_pred"].detach().cpu().numpy()  # (K, Hpred, 2)
                        if action_pred_full.shape[1] < tap_action_chunk:
                            raise ValueError(
                                f"TAP expects action_chunk={tap_action_chunk}, but DP action_pred has horizon={action_pred_full.shape[1]}."
                            )
                        tap_actions_np = action_pred_full[:, :tap_action_chunk, :]
                    tap_actions = torch.from_numpy(tap_actions_np).to(device).float().unsqueeze(0)

                    tap_logits = tap_model(tap_obs, tap_actions)  # (1, K)
                    tap_scores = tap_logits[0].cpu().numpy()  # (K,)
                    tap_pick = int(tap_scores.argmax())
                else:
                    tap_scores = np.zeros((K,), dtype=np.float32)
                    tap_pick = 0

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
                    cont_rng = np.random.default_rng(
                        continuation_perturb_seed(perturb_base_seed, seed, policy_step)
                    )
                    batched_continuation(
                        env_pool[:K], branch_obs_histories, branch_dones,
                        branch_max_rewards, dp_policy, n_obs_steps,
                        n_action_steps, L, device, executor,
                        perturb_type=perturb_type, perturb_prob=perturb_prob,
                        patch_size=patch_size, cont_rng=cont_rng, xy=episode_xy,
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
                    "tap_obs_source": (
                        "perturbed" if (use_tap and tap_sees_perturb)
                        else ("clean" if use_tap else "disabled")
                    ),
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
                if perturb_type is not None:
                    step_rng = np.random.default_rng(
                        decision_perturb_seed(perturb_base_seed, seed, policy_step)
                    )
                    clean_np = obs_for_dp["image"][0].cpu().numpy()
                    perturbed_np = perturb_single_obs(
                        clean_np, perturb_type, step_rng,
                        patch_size=patch_size, prob=perturb_prob,
                        xy=episode_xy,
                    )
                    obs_for_dp["image"] = torch.from_numpy(perturbed_np).to(device).float().unsqueeze(0)
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

    # Changed frac: how often TAP picks something other than candidate 0
    changed = sum(1 for d in all_dps if d["tap_pick"] != 0)
    changed_frac = changed / n if n > 0 else 0.0

    # Fraction of oracle headroom captured (only where headroom > 0)
    has_headroom = headroom > 0.001
    if has_headroom.sum() > 0:
        capture_ratio = tap_gain[has_headroom] / headroom[has_headroom]
        capture_ratio = np.clip(capture_ratio, 0, 1)
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

    # TAP harms: how often does TAP pick worse than candidate 0?
    tap_harms = sum(
        1 for d in all_dps
        if d["reward_tap"] < d["reward_notap"] - 0.001
    )

    # ── Episode-level rollups ──
    # Group decision points by episode and aggregate the decision-point endpoint.
    episodes = defaultdict(list)
    for d in all_dps:
        episodes[d["episode"]].append(d)

    ep_stats = []
    for ep_idx in sorted(episodes.keys()):
        ep_dps = episodes[ep_idx]
        ep_notap_max = max(d["reward_notap"] for d in ep_dps)
        ep_tap_max = max(d["reward_tap"] for d in ep_dps)
        ep_oracle_max = max(d["reward_oracle"] for d in ep_dps)
        ep_notap_mean = float(np.mean([d["reward_notap"] for d in ep_dps]))
        ep_tap_mean = float(np.mean([d["reward_tap"] for d in ep_dps]))
        ep_oracle_mean = float(np.mean([d["reward_oracle"] for d in ep_dps]))
        ep_changed = sum(1 for d in ep_dps if d["tap_pick"] != 0)
        ep_n_dps = len(ep_dps)
        ep_stats.append({
            "episode": ep_idx,
            "seed": ep_dps[0]["seed"],
            "n_decision_points": ep_n_dps,
            "mean_reward_notap": ep_notap_mean,
            "mean_reward_tap": ep_tap_mean,
            "mean_reward_oracle": ep_oracle_mean,
            "max_reward_notap": ep_notap_max,
            "max_reward_tap": ep_tap_max,
            "max_reward_oracle": ep_oracle_max,
            "mean_lift": ep_tap_mean - ep_notap_mean,
            "mean_headroom": ep_oracle_mean - ep_notap_mean,
            "max_lift": ep_tap_max - ep_notap_max,
            "max_headroom": ep_oracle_max - ep_notap_max,
            "changed_frac": ep_changed / ep_n_dps if ep_n_dps > 0 else 0.0,
        })

    ep_mean_notap = np.array([e["mean_reward_notap"] for e in ep_stats], dtype=np.float64)
    ep_mean_tap = np.array([e["mean_reward_tap"] for e in ep_stats], dtype=np.float64)
    ep_mean_oracle = np.array([e["mean_reward_oracle"] for e in ep_stats], dtype=np.float64)
    ep_mean_lifts = np.array([e["mean_lift"] for e in ep_stats], dtype=np.float64)
    ep_mean_headrooms = np.array([e["mean_headroom"] for e in ep_stats], dtype=np.float64)
    ep_mean_has_headroom = ep_mean_headrooms > 0.001
    if ep_mean_has_headroom.sum() > 0:
        ep_mean_captures = ep_mean_lifts[ep_mean_has_headroom] / ep_mean_headrooms[ep_mean_has_headroom]
        ep_mean_captures = np.clip(ep_mean_captures, 0, 1)
        ep_mean_capture = float(ep_mean_captures.mean())
    else:
        ep_mean_capture = 0.0
    n_rescued = int((ep_mean_lifts > 0.001).sum())

    print("\n" + "=" * 60)
    print("RERANKING RESULTS")
    print("=" * 60)

    print(f"\n{'─'*60}")
    print("DECISION-POINT LEVEL")
    print(f"{'─'*60}")
    print(f"  Decision points: {n}")
    print(f"  With headroom (oracle > notap + 0.001): {has_headroom.sum()}/{n}")
    print(f"  Changed frac (TAP picks != 0):  {changed}/{n} ({changed_frac:.1%})")

    print(f"\n  MEAN MAX-STEP REWARD OVER BRANCH HORIZON")
    print(f"  (This is NOT episode coverage/success rate.)")
    print(f"    K_notap (baseline):  {r_notap.mean():.4f}")
    print(f"    K_TAP (reranked):    {r_tap.mean():.4f}")
    print(f"    Oracle best-of-K:    {r_oracle.mean():.4f}")

    print(f"\n  HEADROOM CAPTURE")
    print(f"    Mean TAP gain:              {tap_gain.mean():.4f}")
    print(f"    Mean oracle headroom:       {headroom.mean():.4f}")
    print(f"    Mean capture ratio:         {mean_capture:.1%}")
    print(f"    Median capture ratio:       {median_capture:.1%}")
    print(f"    TAP picks oracle best:      {tap_picks_oracle}/{n} ({tap_picks_oracle/n:.1%})")
    print(f"    TAP beats baseline:         {tap_beats_baseline}/{n} ({tap_beats_baseline/n:.1%})")
    print(f"    TAP harms (picks worse):    {tap_harms}/{n} ({tap_harms/n:.1%})")
    print(f"    Net benefit (helps-harms):  {tap_beats_baseline - tap_harms}/{n}")

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
        print(f"\n    [{label}] (n={mask.sum()})")
        print(f"      Headroom:  {hm.mean():.4f}")
        print(f"      TAP gain:  {tg.mean():.4f}")
        print(f"      Capture:   {cr:.1%}")

    print(f"\n{'─'*60}")
    print("EPISODE LEVEL")
    print(f"{'─'*60}")
    n_eps = len(ep_stats)
    print(f"  Episodes: {n_eps}")
    print(f"  Episodes with headroom: {int(ep_mean_has_headroom.sum())}/{n_eps}")
    print(f"  Episodes rescued (lift > 0): {n_rescued}/{n_eps}")
    ci_notap = mean_ci95(ep_mean_notap)
    ci_tap = mean_ci95(ep_mean_tap)
    ci_oracle = mean_ci95(ep_mean_oracle)
    print("  Mean episode reward (decision-point metric aggregated within episode):")
    print(f"    notap:   {ci_notap['mean']:.4f} [{ci_notap['ci95_low']:.4f}, {ci_notap['ci95_high']:.4f}]")
    print(f"    TAP:     {ci_tap['mean']:.4f} [{ci_tap['ci95_low']:.4f}, {ci_tap['ci95_high']:.4f}]")
    print(f"    oracle:  {ci_oracle['mean']:.4f} [{ci_oracle['ci95_low']:.4f}, {ci_oracle['ci95_high']:.4f}]")
    print(f"  Mean episode lift:     {ep_mean_lifts.mean():.4f}")
    print(f"  Mean episode headroom: {ep_mean_headrooms.mean():.4f}")
    print(f"  Mean episode capture:  {ep_mean_capture:.1%}")

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
            "patch_size": patch_size, "perturb_seed": perturb_base_seed,
            "occlusion_mode": occlusion_mode,
            "use_tap": use_tap,
            "tap_sees_perturb": tap_sees_perturb if use_tap else False,
            "tap_checkpoint": args.tap_checkpoint,
            "tap_config": args.tap_config,
            "perturb_seed_formula_decision": "perturb_seed + seed*10000 + policy_step",
            "perturb_seed_formula_continuation_base": "perturb_seed + seed*10000 + policy_step*100",
            "reward_metric": "max_step_reward_over_candidate_chunk_plus_continuation",
        },
        "summary": {
            "n_decision_points": n,
            "n_with_headroom": int(has_headroom.sum()),
            "changed_frac": changed_frac,
            "mean_reward_notap": float(r_notap.mean()),
            "mean_reward_tap": float(r_tap.mean()),
            "mean_reward_oracle": float(r_oracle.mean()),
            "mean_max_step_reward_notap": float(r_notap.mean()),
            "mean_max_step_reward_tap": float(r_tap.mean()),
            "mean_max_step_reward_oracle": float(r_oracle.mean()),
            "mean_tap_gain": float(tap_gain.mean()),
            "mean_headroom": float(headroom.mean()),
            "mean_capture_ratio": mean_capture,
            "median_capture_ratio": median_capture,
            "tap_picks_oracle_frac": float(tap_picks_oracle / n),
            "tap_beats_baseline_frac": float(tap_beats_baseline / n),
            "tap_harms_frac": float(tap_harms / n),
            "net_benefit": tap_beats_baseline - tap_harms,
        },
        "episode_summary": {
            "n_episodes": n_eps,
            "n_with_headroom": int(ep_mean_has_headroom.sum()),
            "n_rescued": n_rescued,
            "mean_episode_reward_notap": float(ep_mean_notap.mean()),
            "mean_episode_reward_tap": float(ep_mean_tap.mean()),
            "mean_episode_reward_oracle": float(ep_mean_oracle.mean()),
            "mean_episode_reward_notap_ci95": mean_ci95(ep_mean_notap),
            "mean_episode_reward_tap_ci95": mean_ci95(ep_mean_tap),
            "mean_episode_reward_oracle_ci95": mean_ci95(ep_mean_oracle),
            "mean_episode_lift": float(ep_mean_lifts.mean()),
            "mean_episode_headroom": float(ep_mean_headrooms.mean()),
            "mean_episode_capture_ratio": ep_mean_capture,
        },
        "episodes": ep_stats,
        "decision_points": all_dps,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
