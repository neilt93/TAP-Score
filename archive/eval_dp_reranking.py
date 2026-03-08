# /workspace/pennstate-project/eval_dp_reranking.py
"""
TAP-Guided Best-of-K Reranking with Diffusion Policy (Push-T)

At each policy step:
1) Diffusion Policy samples K candidate action chunks
2) TAP scores each candidate (obs, action-chunk)
3) Execute the best candidate (optionally with a margin gate)

Outputs:
- Success rate vs K for each condition
- Timing (DP ms, TAP ms) per policy step
- Changed%: fraction of policy steps where selected_index != 0
- Rescued seeds: seeds where K=1 fails and K>1 succeeds AND reranking actually changed something

Usage (smoke):
python eval_dp_reranking.py \
  --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
  --tap_checkpoint checkpoints_contrastive/pusht/contrastive_tap_h8_best.pt \
  --expert_data data/raw/pusht/pusht_cchi_v7_replay.zarr \
  --k_values 1 4 \
  --n_episodes 5 \
  --perturbations prob_occlusion mild_blur \
  --output_dir eval_results/dp_reranking_smoke \
  --device cuda \
  --batch_size 10 \
  --margin_delta 0.0 \
  --global_seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import dill
import hydra
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from tap.contrastive import ContrastiveTAPScore


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

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


def load_tap_score(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = ContrastiveTAPScore(
        obs_channels=int(cfg.get("obs_channels", 3)),
        action_dim=int(cfg.get("action_dim", 2)),
        obs_window=int(cfg.get("obs_window", 2)),
        action_chunk=int(cfg.get("action_chunk", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        temperature=float(cfg.get("temperature", 0.2)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


# -----------------------------------------------------------------------------
# Expert negative pool
# -----------------------------------------------------------------------------

def build_negative_pool(expert_data_path: str, action_chunk: int, n_samples: int = 2000, seed: int = 0) -> np.ndarray:
    import zarr

    rng = np.random.default_rng(seed)

    root = zarr.open_group(str(expert_data_path), mode="r")
    actions = root["data"]["action"][:]
    episode_ends = root["meta"]["episode_ends"][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    pool: List[np.ndarray] = []
    for _ in range(n_samples * 2):
        if len(pool) >= n_samples:
            break
        ep_idx = rng.integers(0, len(episode_ends))
        s, e = int(episode_starts[ep_idx]), int(episode_ends[ep_idx])
        ep_actions = actions[s:e]
        if len(ep_actions) < action_chunk:
            continue
        t = rng.integers(0, len(ep_actions) - action_chunk + 1)
        pool.append(ep_actions[t : t + action_chunk].astype(np.float32))

    if len(pool) < 16:
        raise RuntimeError(f"Negative pool too small: {len(pool)}")
    return np.stack(pool, axis=0)


# -----------------------------------------------------------------------------
# Perturbations (image is CHW float32 in [0,1])
# -----------------------------------------------------------------------------

def _ensure_float_chw(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
        img = np.transpose(img, (2, 0, 1))
    return img


def normalize_obs_dict(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    if "image" in obs_dict:
        obs_dict["image"] = _ensure_float_chw(obs_dict["image"])
    return obs_dict


def add_noise(img: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    noise = rng.standard_normal(img.shape, dtype=np.float32) * std
    return np.clip(img + noise, 0.0, 1.0)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    # blur spatial dims only for CHW
    return gaussian_filter(img, sigma=(0, sigma, sigma))


def brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img * factor, 0.0, 1.0)


def occlusion(img: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    out = img.copy()
    _, H, W = out.shape
    y = int(rng.integers(0, max(1, H - size)))
    x = int(rng.integers(0, max(1, W - size)))
    out[:, y : y + size, x : x + size] = 0.0
    return out


def get_perturbation_fn(name: Optional[str], rng: np.random.Generator):
    if name is None:
        return None

    if name == "noise":
        return lambda img: add_noise(img, rng, std=0.30)
    if name == "mild_noise":
        return lambda img: add_noise(img, rng, std=0.10)

    if name == "blur":
        return lambda img: blur(img, sigma=2.0)
    if name == "mild_blur":
        return lambda img: blur(img, sigma=1.0)

    if name == "brightness":
        return lambda img: brightness(img, factor=1.5)
    if name == "dark":
        return lambda img: brightness(img, factor=0.5)

    if name == "occlusion":
        return lambda img: occlusion(img, rng, size=20)
    if name == "mild_occlusion":
        return lambda img: occlusion(img, rng, size=12)
    if name == "prob_occlusion":
        # apply occlusion with prob p, otherwise no-op
        p = 0.30
        size = 12
        return lambda img: occlusion(img, rng, size=size) if rng.random() < p else img

    raise ValueError(f"Unknown perturbation: {name}")


class ObservationPerturbationWrapper:
    def __init__(self, env, perturb_fn):
        self.env = env
        self.perturb_fn = perturb_fn

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        normalize_obs_dict(obs)
        if self.perturb_fn is not None:
            obs["image"] = self.perturb_fn(obs["image"])
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        normalize_obs_dict(obs)
        if self.perturb_fn is not None:
            obs["image"] = self.perturb_fn(obs["image"])
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# -----------------------------------------------------------------------------
# TAP scoring utilities
# -----------------------------------------------------------------------------

def score_k_candidates(
    tap_model: ContrastiveTAPScore,
    obs_img_np: np.ndarray,              # (T,C,H,W)
    candidates: np.ndarray,              # (K,H,Da)
    negative_pool: np.ndarray,           # (N,H,Da)
    device: torch.device,
    m_negatives: int,
    neg_seed: int,
) -> Tuple[int, List[float], float]:
    """
    Vectorized scoring:
    For each of K candidates, build [cand, M negatives] and run TAP once over K.
    Negatives are deterministic given neg_seed.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(neg_seed)

    K = int(candidates.shape[0])
    H = int(candidates.shape[1])
    Da = int(candidates.shape[2])

    neg_idx = rng.choice(len(negative_pool), size=m_negatives, replace=False)
    neg_actions = negative_pool[neg_idx]  # (M,H,Da)

    negs_exp = np.broadcast_to(neg_actions[None, :, :, :], (K, m_negatives, H, Da))
    cand_block = np.concatenate([candidates[:, None, :, :], negs_exp], axis=1)  # (K,1+M,H,Da)

    obs_block = np.broadcast_to(obs_img_np[None, :, :, :, :], (K,) + obs_img_np.shape)  # (K,T,C,H,W)

    obs_t = torch.from_numpy(obs_block.copy()).to(device).float()
    cand_t = torch.from_numpy(cand_block.copy()).to(device).float()

    with torch.no_grad():
        logits = tap_model(obs_t, cand_t)  # (K,1+M)
        margins = logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=1)  # (K,)

    scores = margins.detach().cpu().tolist()
    best_idx = int(torch.argmax(margins).item())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return best_idx, scores, elapsed_ms


# -----------------------------------------------------------------------------
# Rollout
# -----------------------------------------------------------------------------

@dataclass
class EnvState:
    env: Any
    obs_dict: Dict[str, Any]
    obs_history: Dict[str, List[np.ndarray]]
    seed: int
    env_steps: int
    done: bool
    rewards: List[float]
    step_data: List[Dict[str, Any]]
    policy_step: int


def _stack_history(hist: List[np.ndarray], T: int) -> np.ndarray:
    if len(hist) < T:
        padded = [hist[0]] * (T - len(hist)) + list(hist)
    else:
        padded = list(hist[-T:])
    return np.stack(padded, axis=0)


def run_reranking_rollouts(
    dp_policy,
    tap_model: Optional[ContrastiveTAPScore],
    negative_pool: np.ndarray,
    K: int,
    seeds: List[int],
    n_action_steps: int,
    n_obs_steps: int,
    tap_action_chunk: int,
    device: torch.device,
    perturbation: Optional[str],
    perturbation_name: str,
    max_env_steps: int,
    legacy: bool,
    success_threshold: float,
    m_negatives: int,
    batch_size: int,
    margin_delta: float,
    global_seed: int,
    use_tap: bool = True,
) -> List[Dict[str, Any]]:
    completed: List[Dict[str, Any]] = []
    batch_size = min(batch_size, len(seeds))
    next_idx = 0

    def make_state(seed: int) -> EnvState:
        env = PushTImageEnv(legacy=legacy, render_size=96)
        if perturbation is not None:
            env_rng = np.random.default_rng(seed)
            pfn = get_perturbation_fn(perturbation, env_rng)
            env = ObservationPerturbationWrapper(env, pfn)

        env.seed(seed)
        obs = env.reset()
        normalize_obs_dict(obs)

        hist = defaultdict(list)
        for k, v in obs.items():
            hist[k].append(v)

        return EnvState(
            env=env,
            obs_dict=obs,
            obs_history=hist,
            seed=seed,
            env_steps=0,
            done=False,
            rewards=[],
            step_data=[],
            policy_step=0,
        )

    active: List[EnvState] = []
    for _ in range(batch_size):
        active.append(make_state(seeds[next_idx]))
        next_idx += 1

    pbar = tqdm(total=len(seeds), desc=f"Rollouts K={K} ({perturbation_name})")

    diversity_logged = False
    first_shapes_logged = False

    while active:
        B = len(active)

        # Build batched obs tensors for DP over all keys
        batched_obs: Dict[str, List[np.ndarray]] = defaultdict(list)
        batched_img_np: List[np.ndarray] = []

        for st in active:
            for key in st.obs_dict.keys():
                batched_obs[key].append(_stack_history(st.obs_history[key], n_obs_steps))
            batched_img_np.append(_stack_history(st.obs_history["image"], n_obs_steps))

        obs_tensor = {k: torch.from_numpy(np.stack(v, axis=0)).to(device).float() for k, v in batched_obs.items()}

        if not first_shapes_logged:
            for key, ten in obs_tensor.items():
                print(f"  obs_tensor['{key}']: {tuple(ten.shape)}")
            first_shapes_logged = True

        # DP sampling
        if K == 1:
            t0 = time.perf_counter()
            with torch.no_grad():
                action_dict = dp_policy.predict_action(obs_tensor)
            dp_ms_per_env = (time.perf_counter() - t0) * 1000.0 / B
            all_actions = action_dict["action"].detach().cpu().numpy()  # (B, n_action_steps, Da)

            selected_indices = [0] * B
            selected_actions = all_actions  # (B, n_action_steps, Da)
            tap_scores_per_env: List[List[float]] = [[] for _ in range(B)]
            tap_ms_per_env = 0.0

        else:
            obs_expanded = {k: v.repeat_interleave(K, dim=0) for k, v in obs_tensor.items()}  # (B*K, ...)
            t0 = time.perf_counter()
            with torch.no_grad():
                action_dict = dp_policy.predict_action(obs_expanded)
            dp_ms_per_env = (time.perf_counter() - t0) * 1000.0 / B

            flat = action_dict["action"].detach().cpu().numpy()  # (B*K, n_action_steps, Da)
            all_actions = flat.reshape(B, K, n_action_steps, -1)

            if not diversity_logged:
                diversity_logged = True
                dists = []
                for i in range(K):
                    for j in range(i + 1, K):
                        dists.append(np.linalg.norm(all_actions[0, i] - all_actions[0, j], axis=-1).mean())
                print(f"  K={K} candidate diversity: mean pairwise L2 = {float(np.mean(dists)):.4f}")

            # Prepare candidates for TAP scoring: (B,K,H,Da)
            if n_action_steps >= tap_action_chunk:
                tap_cands = all_actions[:, :, :tap_action_chunk, :]
            else:
                pad = tap_action_chunk - n_action_steps
                pad_block = np.tile(all_actions[:, :, -1:, :], (1, 1, pad, 1))
                tap_cands = np.concatenate([all_actions, pad_block], axis=2)

            # Score each env (or skip TAP for no-TAP control)
            selected_indices = []
            selected_actions = np.zeros((B, n_action_steps, all_actions.shape[-1]), dtype=np.float32)

            if use_tap and tap_model is not None:
                t0_tap = time.perf_counter()
                margins_bk = np.zeros((B, K), dtype=np.float32)
                scores_b: List[List[float]] = []

                for bi, st in enumerate(active):
                    neg_seed = int(global_seed + st.seed * 100000 + st.policy_step)
                    best_idx, scores_k, _ = score_k_candidates(
                        tap_model=tap_model,
                        obs_img_np=batched_img_np[bi],
                        candidates=tap_cands[bi],
                        negative_pool=negative_pool,
                        device=device,
                        m_negatives=m_negatives,
                        neg_seed=neg_seed,
                    )
                    scores_b.append(scores_k)
                    margins_bk[bi] = np.array(scores_k, dtype=np.float32)

                tap_ms_per_env = (time.perf_counter() - t0_tap) * 1000.0 / B
                tap_scores_per_env = scores_b

                for bi in range(B):
                    scores_k = margins_bk[bi]
                    best_idx = int(np.argmax(scores_k))
                    if margin_delta > 0.0 and best_idx != 0:
                        if float(scores_k[best_idx] - scores_k[0]) < margin_delta:
                            best_idx = 0
                    selected_indices.append(best_idx)
                    selected_actions[bi] = all_actions[bi, best_idx]
            else:
                # No-TAP control: always pick candidate 0
                tap_ms_per_env = 0.0
                tap_scores_per_env = [[] for _ in range(B)]
                for bi in range(B):
                    selected_indices.append(0)
                    selected_actions[bi] = all_actions[bi, 0]

        # Step envs
        finished: List[int] = []
        for bi, st in enumerate(active):
            st.step_data.append(
                {
                    "selected_index": int(selected_indices[bi]),
                    "all_tap_scores": tap_scores_per_env[bi],
                    "dp_time_ms": float(dp_ms_per_env),
                    "tap_time_ms": float(tap_ms_per_env),
                }
            )

            # Execute chunk
            for act in selected_actions[bi]:
                obs, reward, done, _info = st.env.step(act)
                normalize_obs_dict(obs)
                for key, val in obs.items():
                    st.obs_history[key].append(val)
                st.rewards.append(float(reward))
                st.env_steps += 1

                if done or st.env_steps >= max_env_steps:
                    st.done = True
                    break

            st.obs_dict = obs
            st.policy_step += 1

            if st.done:
                finished.append(bi)

        # Collect finished, refill
        for idx in sorted(finished, reverse=True):
            st = active.pop(idx)
            rewards_arr = np.array(st.rewards, dtype=np.float32)
            max_r = float(np.max(rewards_arr)) if rewards_arr.size else 0.0
            success = bool(max_r >= success_threshold)

            rerank_changed = [s["selected_index"] != 0 for s in st.step_data]
            changed_frac = float(np.mean(rerank_changed)) if rerank_changed else 0.0
            changed_any = bool(any(rerank_changed))

            completed.append(
                {
                    "seed": int(st.seed),
                    "K": int(K),
                    "perturbation": perturbation_name,
                    "success": bool(success),
                    "max_reward": float(max_r),
                    "mean_reward": float(np.mean(rewards_arr)) if rewards_arr.size else 0.0,
                    "n_steps": int(len(st.step_data)),
                    "rerank_changed_frac": float(changed_frac),
                    "changed_any": changed_any,
                    "mean_dp_ms": float(np.mean([s["dp_time_ms"] for s in st.step_data])) if st.step_data else 0.0,
                    "mean_tap_ms": float(np.mean([s["tap_time_ms"] for s in st.step_data])) if st.step_data else 0.0,
                    "step_data": st.step_data,
                }
            )
            pbar.update(1)

            if next_idx < len(seeds):
                active.insert(idx, make_state(seeds[next_idx]))
                next_idx += 1

    pbar.close()
    return completed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TAP-Guided Best-of-K Reranking (Push-T)")
    parser.add_argument("--dp_checkpoint", type=str, required=True)
    parser.add_argument("--tap_checkpoint", type=str, required=True)
    parser.add_argument("--expert_data", type=str, required=True)

    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument(
        "--perturbations",
        type=str,
        nargs="*",
        default=[],
        choices=[
            "noise", "mild_noise",
            "blur", "mild_blur",
            "brightness", "dark",
            "occlusion", "mild_occlusion", "prob_occlusion",
        ],
    )
    parser.add_argument("--m_negatives", type=int, default=15)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--success_threshold", type=float, default=0.80)
    parser.add_argument("--margin_delta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_env_steps", type=int, default=200)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--include_notap_control", action="store_true",
                        help="For K>1, also run a no-TAP control (always pick candidate 0) to isolate RNG effects")
    parser.add_argument("--output_dir", type=str, default="eval_results/dp_reranking")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TAP-Guided Best-of-K Reranking with Diffusion Policy")
    print("=" * 70)
    print(f"DP checkpoint:  {args.dp_checkpoint}")
    print(f"TAP checkpoint: {args.tap_checkpoint}")
    print(f"Expert data:    {args.expert_data}")
    print(f"K values:       {args.k_values}")
    print(f"Episodes/combo: {args.n_episodes}")
    print(f"Perturbations:  {args.perturbations if args.perturbations else ['clean only']}")
    print(f"Margin delta:   {args.margin_delta}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Device:         {args.device}")
    print("=" * 70)

    # Load models
    print("\n[1/4] Loading models...")
    dp_policy, dp_cfg = load_diffusion_policy(args.dp_checkpoint, device)
    tap_model, tap_cfg = load_tap_score(args.tap_checkpoint, device)

    n_action_steps = int(dp_cfg.task.env_runner.n_action_steps)
    n_obs_steps = int(dp_cfg.task.env_runner.n_obs_steps)
    tap_action_chunk = int(tap_cfg.get("action_chunk", 8))

    print(f"  DP: n_action_steps={n_action_steps}, n_obs_steps={n_obs_steps}")
    print(f"  TAP: action_chunk={tap_action_chunk}")

    # Negative pool
    print("\n[2/4] Building negative action pool...")
    negative_pool = build_negative_pool(args.expert_data, action_chunk=tap_action_chunk, n_samples=2000)

    # Seeds
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_episodes))
    conditions = ["clean"] + list(args.perturbations)

    all_results: List[Dict[str, Any]] = []

    # Build list of (condition, K, use_tap, label) runs
    runs: List[Tuple[str, Optional[str], int, bool, str]] = []
    for cond in conditions:
        pert = None if cond == "clean" else cond
        for K in args.k_values:
            runs.append((cond, pert, K, True, cond))
            # Add no-TAP control for K>1
            if args.include_notap_control and K > 1:
                runs.append((cond, pert, K, False, f"{cond}_notap"))

    print(f"\n[3/4] Running reranking rollouts ({len(runs)} combos)...")
    for cond, pert, K, use_tap, label in runs:
        tap_tag = "" if use_tap else " [no-TAP control]"
        print(f"\n--- Condition: {label}, K={K}{tap_tag} ---")
        eps = run_reranking_rollouts(
            dp_policy=dp_policy,
            tap_model=tap_model if K > 1 else None,
            negative_pool=negative_pool,
            K=K,
            seeds=seeds,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            tap_action_chunk=tap_action_chunk,
            device=device,
            perturbation=pert,
            perturbation_name=label,
            max_env_steps=args.max_env_steps,
            legacy=args.legacy,
            success_threshold=args.success_threshold,
            m_negatives=args.m_negatives,
            batch_size=args.batch_size,
            margin_delta=args.margin_delta,
            global_seed=args.global_seed,
            use_tap=use_tap,
        )
        all_results.extend(eps)

        sr = float(np.mean([e["success"] for e in eps])) if eps else 0.0
        mr = float(np.mean([e["max_reward"] for e in eps])) if eps else 0.0
        ch = float(np.mean([e["rerank_changed_frac"] for e in eps])) if eps else 0.0
        print(f"  Success: {sr:.1%}, Mean reward: {mr:.3f}, Changed%: {ch:.1%}")

    # Summary
    print("\n[4/4] Computing summary...")

    # Collect all unique labels from runs
    all_labels = list(dict.fromkeys(label for _, _, _, _, label in runs))

    summary_rows: List[Dict[str, Any]] = []
    for label in all_labels:
        for K in sorted(set(k for _, _, k, _, l in runs if l == label)):
            eps = [e for e in all_results if e["perturbation"] == label and e["K"] == K]
            if not eps:
                continue
            summary_rows.append(
                {
                    "condition": label,
                    "K": K,
                    "success_rate": float(np.mean([e["success"] for e in eps])),
                    "mean_reward": float(np.mean([e["max_reward"] for e in eps])),
                    "dp_ms": float(np.mean([e["mean_dp_ms"] for e in eps])),
                    "tap_ms": float(np.mean([e["mean_tap_ms"] for e in eps])),
                    "total_ms": float(np.mean([e["mean_dp_ms"] + e["mean_tap_ms"] for e in eps])),
                    "changed_frac": float(np.mean([e["rerank_changed_frac"] for e in eps])),
                    "n_episodes": len(eps),
                }
            )

    print("\n" + "=" * 90)
    print("SUMMARY: Best-of-K Reranking Results")
    print("=" * 90)
    print(f"{'Condition':>14}  {'K':>3}  {'Success%':>8}  {'Reward':>7}  {'DP ms':>7}  {'TAP ms':>7}  {'Total ms':>8}  {'Changed%':>8}")
    print("-" * 90)
    for row in summary_rows:
        print(
            f"{row['condition']:>14}  {row['K']:>3}  {row['success_rate']:>7.1%}  {row['mean_reward']:>7.3f}  "
            f"{row['dp_ms']:>7.1f}  {row['tap_ms']:>7.1f}  {row['total_ms']:>8.1f}  {row['changed_frac']:>7.1%}"
        )
    print("=" * 90)

    # Rescued seeds: require changed_any=True (TAP actually intervened)
    # Only compare TAP runs (not _notap controls) against their K=1 baseline
    rescued: Dict[str, Dict[str, List[int]]] = {}
    for cond in conditions:
        k1 = {e["seed"]: e for e in all_results if e["perturbation"] == cond and e["K"] == 1}
        if not k1:
            continue
        rescued[cond] = {}
        for K in args.k_values:
            if K == 1:
                continue
            kN = {e["seed"]: e for e in all_results if e["perturbation"] == cond and e["K"] == K}
            seeds_rescued = []
            for seed, ep1 in k1.items():
                if seed not in kN:
                    continue
                epN = kN[seed]
                if (not ep1["success"]) and epN["success"] and epN.get("changed_any", False):
                    seeds_rescued.append(int(seed))
            if seeds_rescued:
                rescued[cond][str(K)] = seeds_rescued

    if any(rescued.values()):
        print("\nRESCUED EPISODES (failed at K=1, succeeded at K>1, changed_any=True):")
        for cond, byK in rescued.items():
            for k_str, seeds_list in byK.items():
                print(f"  {cond} K={k_str}: {len(seeds_list)} seeds (example: {seeds_list[:5]})")

    # Save JSON
    results_json = {
        "config": {
            "dp_checkpoint": args.dp_checkpoint,
            "tap_checkpoint": args.tap_checkpoint,
            "expert_data": args.expert_data,
            "k_values": args.k_values,
            "n_episodes": args.n_episodes,
            "perturbations": args.perturbations,
            "m_negatives": args.m_negatives,
            "seed_offset": args.seed_offset,
            "n_action_steps": n_action_steps,
            "n_obs_steps": n_obs_steps,
            "tap_action_chunk": tap_action_chunk,
            "success_threshold": args.success_threshold,
            "margin_delta": args.margin_delta,
            "batch_size": args.batch_size,
            "global_seed": args.global_seed,
        },
        "summary": summary_rows,
        "rescued_episodes": rescued,
        "episodes": [
            {
                "seed": e["seed"],
                "K": e["K"],
                "perturbation": e["perturbation"],
                "success": bool(e["success"]),
                "max_reward": float(e["max_reward"]),
                "mean_reward": float(e["mean_reward"]),
                "n_steps": int(e["n_steps"]),
                "rerank_changed_frac": float(e["rerank_changed_frac"]),
                "changed_any": bool(e["changed_any"]),
                "mean_dp_ms": float(e["mean_dp_ms"]),
                "mean_tap_ms": float(e["mean_tap_ms"]),
            }
            for e in all_results
        ],
    }

    json_path = out_dir / "reranking_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2, allow_nan=False)
    print(f"\nResults saved to {json_path}")

    # Save traces to NPZ
    npz_dict: Dict[str, Any] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in all_results:
        grouped[f"{e['perturbation']}_K{e['K']}"].append(e)

    for key, eps in grouped.items():
        for i, ep in enumerate(eps):
            prefix = f"{key}_ep{i}"
            npz_dict[f"{prefix}_seed"] = np.array(ep["seed"])
            npz_dict[f"{prefix}_success"] = np.array(ep["success"])
            npz_dict[f"{prefix}_max_reward"] = np.array(ep["max_reward"])
            npz_dict[f"{prefix}_selected_indices"] = np.array([s["selected_index"] for s in ep["step_data"]], dtype=np.int32)
            for si, s in enumerate(ep["step_data"]):
                if s["all_tap_scores"]:
                    npz_dict[f"{prefix}_scores_step{si}"] = np.array(s["all_tap_scores"], dtype=np.float32)

    npz_path = out_dir / "reranking_traces.npz"
    np.savez_compressed(npz_path, **npz_dict)
    print(f"Trace data saved to {npz_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
