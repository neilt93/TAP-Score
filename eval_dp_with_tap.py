"""
Diffusion Policy + TAP-Score Integration

Evaluates Diffusion Policy rollouts with TAP-Score detection to demonstrate
that TAP-Score can detect REAL policy failures in closed-loop execution,
not just simulated action corruptions.

Pipeline:
1. Load trained Diffusion Policy checkpoint
2. Load trained TAP-Score checkpoint
3. Run clean + perturbed rollouts using Diffusion Policy
4. Score all rollout trajectories with TAP-Score
5. Report detection metrics (AUROC, TPR@FPR)

Usage:
    python eval_dp_with_tap.py \
        --dp_checkpoint baselines/diffusion_policy/outputs/checkpoint.ckpt \
        --tap_checkpoint checkpoints_contrastive/pusht/contrastive_tap_best.pt \
        --n_clean_rollouts 50 \
        --n_perturbed_rollouts 50
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import dill
import hydra
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.ndimage import gaussian_filter

# Diffusion Policy imports
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.common.pytorch_util import dict_apply

# TAP-Score imports
from tap.contrastive import ContrastiveTAPScore


# ============================================================================
# Model Loading
# ============================================================================

def load_diffusion_policy(checkpoint_path, device):
    """Load Diffusion Policy using Hydra workspace pattern."""
    print(f"Loading Diffusion Policy from {checkpoint_path}...")

    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # Instantiate workspace (handle different config layouts)
    # OmegaConf DictConfig doesn't support .get() — use getattr with defaults
    target = getattr(cfg, '_target_', None)
    if target is None:
        workspace_cfg = getattr(cfg, 'workspace', None)
        if workspace_cfg is not None:
            target = getattr(workspace_cfg, '_target_', None)
    if target is None:
        raise ValueError("Cannot find workspace _target_ in DP config. Check checkpoint format.")
    cls = hydra.utils.get_class(target)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Get policy (EMA if available)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(device)
    policy.eval()

    print("✓ Diffusion Policy loaded successfully")
    return policy, cfg


def load_tap_score(checkpoint_path, device):
    """Load TAP-Score model."""
    print(f"Loading TAP-Score from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = ContrastiveTAPScore(
        obs_channels=config.get("obs_channels", 3),
        action_dim=config.get("action_dim", 2),
        obs_window=config.get("obs_window", 2),
        action_chunk=config.get("action_chunk", 16),
        hidden_dim=config.get("hidden_dim", 128),
        temperature=config.get("temperature", 0.1),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("✓ TAP-Score loaded successfully")
    return model, config


# ============================================================================
# Perturbation Functions
# ============================================================================

def _ensure_4d(obs):
    """Ensure obs is 4D (T, C, H, W). Returns (obs_4d, was_3d)."""
    if obs.ndim == 3:
        return obs[np.newaxis], True
    if obs.ndim == 4:
        return obs, False
    raise ValueError(f"Unexpected image ndim={obs.ndim}, shape={obs.shape}")


def _restore_dim(obs, was_3d):
    """Squeeze back to 3D if input was 3D."""
    return obs[0] if was_3d else obs


def add_gaussian_noise(obs, rng, std=0.3):
    """Add Gaussian noise to observation images. Accepts (C,H,W) or (T,C,H,W)."""
    obs, was_3d = _ensure_4d(obs)
    noise = rng.standard_normal(obs.shape, dtype=np.float32) * std
    result = np.clip(obs + noise, 0, 1)
    return _restore_dim(result, was_3d)


def apply_gaussian_blur(obs, rng, sigma=2.0):
    """Apply Gaussian blur. Accepts (C,H,W) or (T,C,H,W)."""
    if obs.ndim == 3:
        # (C, H, W) — blur only spatial dims
        return gaussian_filter(obs, sigma=(0, sigma, sigma))
    # (T, C, H, W) — blur spatial dims per frame
    blurred = obs.copy()
    for t in range(obs.shape[0]):
        blurred[t] = gaussian_filter(obs[t], sigma=(0, sigma, sigma))
    return blurred


def adjust_brightness(obs, rng, factor=1.5):
    """Adjust brightness. Accepts (C,H,W) or (T,C,H,W)."""
    return np.clip(obs * factor, 0, 1)


def apply_contrast_shift(obs, rng, factor=1.5):
    """Adjust contrast. Accepts (C,H,W) or (T,C,H,W)."""
    mean = obs.mean()
    return np.clip((obs - mean) * factor + mean, 0, 1)


def apply_occlusion(obs, rng, occlusion_size=20):
    """Apply random rectangular occlusion. Accepts (C,H,W) or (T,C,H,W)."""
    obs, was_3d = _ensure_4d(obs)
    occluded = obs.copy()
    H, W = obs.shape[2], obs.shape[3]
    y = rng.integers(0, H - occlusion_size)
    x = rng.integers(0, W - occlusion_size)
    occluded[:, :, y:y+occlusion_size, x:x+occlusion_size] = 0
    return _restore_dim(occluded, was_3d)


def get_perturbation_fn(perturbation_type, rng):
    """Get perturbation function by name. Closures capture per-env rng."""
    perturbations = {
        'noise': lambda obs: add_gaussian_noise(obs, rng, std=0.3),
        'blur': lambda obs: apply_gaussian_blur(obs, rng, sigma=2.0),
        'brightness': lambda obs: adjust_brightness(obs, rng, factor=1.5),
        'dark': lambda obs: adjust_brightness(obs, rng, factor=0.5),
        'contrast': lambda obs: apply_contrast_shift(obs, rng, factor=1.5),
        'occlusion': lambda obs: apply_occlusion(obs, rng, occlusion_size=20),
    }
    return perturbations.get(perturbation_type, None)


# ============================================================================
# Image Normalization
# ============================================================================

def normalize_image(img):
    """Ensure image is float32 in [0, 1] and CHW layout.

    Handles:
    - uint8 [0, 255] → float32 [0, 1]
    - HWC → CHW conversion (if last dim is 3 and first dim != 3)
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)
    # If HWC layout (H, W, 3), convert to CHW (3, H, W)
    if img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] != 3:
        img = np.transpose(img, (2, 0, 1))
    return img


def normalize_obs_dict(obs_dict):
    """Normalize image in obs_dict in-place."""
    if 'image' in obs_dict:
        obs_dict['image'] = normalize_image(obs_dict['image'])
    return obs_dict


# ============================================================================
# Rollout Execution
# ============================================================================

class ObservationPerturbationWrapper:
    """Wrapper that applies perturbations to observations during rollout.

    Applies perturbation in both reset() and step() to ensure consistent
    corruption throughout the episode.
    """

    def __init__(self, env, perturbation_fn=None, rng=None):
        self.env = env
        self.perturbation_fn = perturbation_fn
        self._rng = rng  # per-env RNG, never touches global state

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs)
        normalize_obs_dict(obs_dict)  # Ensure float32 [0,1] CHW before perturbation
        if self.perturbation_fn:
            obs_dict['image'] = self.perturbation_fn(obs_dict['image'])
        return obs_dict

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        normalize_obs_dict(obs_dict)  # Ensure float32 [0,1] CHW before perturbation
        if self.perturbation_fn:
            obs_dict['image'] = self.perturbation_fn(obs_dict['image'])
        return obs_dict, reward, done, info

    def __getattr__(self, name):
        # Delegate all other attributes to wrapped env
        return getattr(self.env, name)


def run_rollouts(policy, n_episodes=None, perturbation_type=None, perturbation_name='clean',
                 seeds=None, seed_offset=0, max_env_steps=200, n_action_steps=8, n_obs_steps=2,
                 device='cuda', batch_size=10, legacy=False, success_threshold=0.80):
    """Run batched closed-loop rollouts with Diffusion Policy.

    Runs up to batch_size environments in parallel, batching DP inference
    for higher GPU utilization. Envs that finish early are replaced from
    the remaining queue until all n_episodes are complete.

    Args:
        policy: Trained Diffusion Policy model
        n_episodes: Number of episodes to run (ignored if seeds is provided)
        perturbation_type: Name of perturbation (e.g. 'noise', 'blur') or None for clean
        perturbation_name: Name of perturbation for logging
        seeds: Explicit list of seeds for paired evaluation. Overrides n_episodes/seed_offset.
        seed_offset: Seed offset for reproducibility (used only if seeds is None)
        max_env_steps: Maximum environment steps per episode
        n_action_steps: Number of action steps to execute per prediction
        n_obs_steps: Number of observation frames in history window
        device: Device for policy inference
        batch_size: Number of parallel environments for batched inference

    Returns:
        List of episode dicts with keys:
            - observations: (T, n_obs_steps, 3, 96, 96) stacked image windows
            - proposed_actions: (T, H, action_dim) full action chunks proposed by policy
            - executed_actions: (T, n_action_steps, action_dim) actions actually executed
            - rewards: per-env-step rewards
            - success: bool indicating if episode succeeded
            - perturbation: str name of perturbation applied
    """
    # Resolve seeds: explicit list or auto-generate from n_episodes + offset
    if seeds is not None:
        episode_seeds = list(seeds)
    else:
        assert n_episodes is not None, "Must provide either seeds or n_episodes"
        episode_seeds = list(range(seed_offset, seed_offset + n_episodes))

    n_total = len(episode_seeds)
    completed_episodes = []
    batch_size = min(batch_size, n_total)

    # Track which episode index (into episode_seeds) to launch next
    next_episode_idx = 0

    def _make_env_state(ep_idx):
        """Create and reset a single environment, return its state dict."""
        seed = episode_seeds[ep_idx]
        env = PushTImageEnv(legacy=legacy, render_size=96)
        if perturbation_type:
            # Per-env RNG: deterministic but independent across envs
            env_rng = np.random.default_rng(seed)
            perturb_fn = get_perturbation_fn(perturbation_type, env_rng)
            env = ObservationPerturbationWrapper(env, perturb_fn, rng=env_rng)
        env.seed(seed)
        obs_dict = env.reset()
        normalize_obs_dict(obs_dict)  # Ensure float32 [0,1] CHW
        obs_history = defaultdict(list)
        for k, v in obs_dict.items():
            obs_history[k].append(v)
        episode = {
            'observations': [],
            'proposed_actions': [],
            'executed_actions': [],
            'rewards': [],
            'seed': seed,
            'perturbation': perturbation_name,
        }
        return {
            'env': env,
            'obs_dict': obs_dict,
            'obs_history': obs_history,
            'episode': episode,
            'done': False,
            'steps': 0,       # counts policy steps
            'env_steps': 0,   # counts environment steps
        }

    # Initialize first batch of environments
    active = []
    for _ in range(batch_size):
        active.append(_make_env_state(next_episode_idx))
        next_episode_idx += 1

    pbar = tqdm(total=n_total, desc=f"Rollouts ({perturbation_name})")

    while active:
        # --- Build batched observation tensor ---
        # Stack observations from all active envs: each is (n_obs_steps, C, H, W)
        batched_obs = defaultdict(list)
        for state in active:
            for k in state['obs_dict'].keys():
                hist = state['obs_history'][k]
                if len(hist) < n_obs_steps:
                    padded = [hist[0]] * (n_obs_steps - len(hist)) + list(hist)
                else:
                    padded = list(hist[-n_obs_steps:])
                batched_obs[k].append(np.stack(padded, axis=0))

        # Convert to tensors: (B, n_obs_steps, ...)
        batched_obs_tensor = {
            k: torch.from_numpy(np.stack(v, axis=0)).to(device).float()
            for k, v in batched_obs.items()
        }

        # --- Batched DP inference ---
        with torch.no_grad():
            action_dict = policy.predict_action(batched_obs_tensor)

        # action_dict['action'] is (B, H, action_dim)
        all_actions = action_dict['action'].detach().cpu().numpy()

        # --- Step each environment ---
        finished_indices = []
        for i, state in enumerate(active):
            full_action_chunk = all_actions[i]  # (H, 2)

            # Store stacked obs for TAP scoring
            hist = state['obs_history']['image']
            if len(hist) < n_obs_steps:
                padded = [hist[0]] * (n_obs_steps - len(hist)) + list(hist)
            else:
                padded = list(hist[-n_obs_steps:])
            stacked_image = np.stack(padded, axis=0)

            state['episode']['observations'].append(stacked_image)
            state['episode']['proposed_actions'].append(full_action_chunk)

            # Execute first n_action_steps
            executed_action = full_action_chunk[:n_action_steps]

            # Step env one action at a time, updating obs history EVERY step
            # so DP always sees consecutive frames
            actually_executed = []
            for act in executed_action:
                obs_dict, reward, done, info = state['env'].step(act)
                normalize_obs_dict(obs_dict)
                for k, v in obs_dict.items():
                    state['obs_history'][k].append(v)
                state['episode']['rewards'].append(reward)
                actually_executed.append(act)
                state['env_steps'] += 1
                if done or state['env_steps'] >= max_env_steps:
                    break

            # Only record actions that were actually executed (truncated on early done)
            state['episode']['executed_actions'].append(np.array(actually_executed))

            state['obs_dict'] = obs_dict
            state['steps'] += 1  # policy steps
            state['done'] = done or state['env_steps'] >= max_env_steps

            # Check if episode is finished
            if state['done']:
                finished_indices.append(i)

        # --- Collect finished episodes, replace with new ones ---
        for i in sorted(finished_indices, reverse=True):
            state = active.pop(i)
            ep = state['episode']
            ep['success'] = np.max(ep['rewards']) >= success_threshold
            ep['max_reward'] = float(np.max(ep['rewards']))
            ep['mean_reward'] = float(np.mean(ep['rewards']))
            ep['observations'] = np.array(ep['observations'])
            ep['proposed_actions'] = np.array(ep['proposed_actions'])
            # executed_actions may have variable length (truncated on early done)
            # so store as list of arrays if lengths differ
            exec_lens = [len(a) for a in ep['executed_actions']]
            if len(set(exec_lens)) == 1:
                ep['executed_actions'] = np.array(ep['executed_actions'])
            else:
                ep['executed_actions'] = ep['executed_actions']  # keep as list of arrays
            ep['rewards'] = np.array(ep['rewards'])
            completed_episodes.append(ep)
            pbar.update(1)

            # Launch replacement env if episodes remain
            if next_episode_idx < n_total:
                active.insert(i, _make_env_state(next_episode_idx))
                next_episode_idx += 1

    pbar.close()
    return completed_episodes


# ============================================================================
# TAP-Score Evaluation
# ============================================================================

def build_negative_pool(expert_data_path, action_chunk, n_samples=2000):
    """Build negative action pool from expert demonstrations.

    This reuses the same negative pool used during TAP training.
    """
    print("Building negative action pool from expert demos...")

    import zarr
    root = zarr.open_group(str(expert_data_path), mode='r')
    actions = root['data']['action'][:]
    episode_ends = root['meta']['episode_ends'][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    pool = []
    for _ in range(n_samples):
        # Sample random episode
        ep_idx = np.random.randint(len(episode_ends))
        start, end = episode_starts[ep_idx], episode_ends[ep_idx]
        ep_actions = actions[start:end]

        if len(ep_actions) < action_chunk:
            continue

        # Sample random chunk
        t = np.random.randint(0, len(ep_actions) - action_chunk)
        pool.append(ep_actions[t:t + action_chunk].astype(np.float32))

    assert len(pool) >= 16, f"Negative pool too small ({len(pool)}), need at least 16 for m_negatives=15"
    print(f"✓ Built negative pool with {len(pool)} action chunks")
    return np.stack(pool)


def score_action_logmargin(model, obs, action, negative_pool, device, m_negatives=15):
    """Score (obs, action) pair using TAP log-margin scoring.

    Returns:
        Log-margin score: log P(action | obs) - logsumexp(log P(neg | obs))
        Higher score = more consistent with expert manifold
    """
    # Sample negatives
    neg_indices = np.random.choice(len(negative_pool), m_negatives, replace=False)
    neg_actions = negative_pool[neg_indices]

    # Construct candidates: [action, neg1, neg2, ..., negM]
    candidates = np.concatenate([action[np.newaxis], neg_actions], axis=0)

    # Convert to tensors
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device).float()       # Fix #6
    candidates_t = torch.from_numpy(candidates).unsqueeze(0).to(device).float()  # Fix #6

    with torch.no_grad():
        logits = model(obs_t, candidates_t)  # (1, M+1)
        l0 = logits[0, 0]  # Log-prob of action
        l_neg = logits[0, 1:]  # Log-probs of negatives
        margin = l0 - torch.logsumexp(l_neg, dim=0)

    return margin.item()


def score_trajectory_with_tap(tap_model, episode, negative_pool, config, device):
    """Score all (obs, proposed_action) pairs in a trajectory.

    CRITICAL: Scores the PROPOSED action chunks from the policy, not executed actions.
    This tests whether TAP can detect when the policy proposes off-manifold actions.

    Returns:
        Array of TAP scores (one per timestep)
    """
    obs_window = config.get('obs_window', 2)
    action_chunk = config.get('action_chunk', 16)

    observations = episode['observations']  # (T, 2, 3, 96, 96)
    proposed_actions = episode['proposed_actions']  # (T, H, 2) where H is DP horizon

    scores = []
    timesteps = []

    # Score each timestep where we have a proposal
    for t in range(len(observations)):
        # Get observation (already in correct format from env)
        obs = observations[t]  # (2, 3, 96, 96)

        # Get proposed action chunk
        proposed = proposed_actions[t]  # (H, 2) where H is DP horizon

        # Construct action chunk for TAP (needs action_chunk steps)
        # If TAP action_chunk matches DP horizon (e.g. both 8), no stitching needed
        if len(proposed) >= action_chunk:
            # Direct match — use proposal directly (normal case when horizons match)
            action_chunk_data = proposed[:action_chunk]
        elif len(proposed) >= action_chunk // 2:
            # Stitching fallback if DP horizon < TAP action_chunk
            # Concatenate with next proposal if available
            if t + 1 < len(proposed_actions):
                next_proposed = proposed_actions[t + 1]
                action_chunk_data = np.vstack([proposed, next_proposed])[:action_chunk]
            else:
                # Pad with last action
                pad_length = action_chunk - len(proposed)
                action_chunk_data = np.vstack([
                    proposed,
                    np.tile(proposed[-1:], (pad_length, 1))
                ])
        else:
            # Skip if proposal is too short
            continue

        # Ensure correct shape (action_chunk, action_dim)
        action_dim = proposed.shape[1]  # Should be 2 for Push-T
        assert action_chunk_data.shape == (action_chunk, action_dim), \
            f"Expected ({action_chunk}, {action_dim}), got {action_chunk_data.shape}"

        # Score with TAP
        score = score_action_logmargin(
            tap_model, obs, action_chunk_data, negative_pool, device, m_negatives=15
        )
        scores.append(score)
        timesteps.append(t)

    return np.array(scores), np.array(timesteps)


def aggregate_episode_score(scores, method='percentile_10'):
    """Aggregate timestep scores to episode-level score.

    Args:
        scores: Array of timestep scores
        method: Aggregation method ('percentile_10', 'min', 'mean')

    Returns:
        Single episode score
    """
    if len(scores) == 0:
        return 0.0

    if method == 'percentile_10':
        return np.percentile(scores, 10)
    elif method == 'min':
        return np.min(scores)
    elif method == 'mean':
        return np.mean(scores)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ============================================================================
# Metrics Computation
# ============================================================================

def first_flag(tap_scores, timesteps, threshold):
    """Find first timestep where TAP score drops below threshold.

    Returns:
        (idx, timestep) — index into tap_scores and corresponding timestep.
        (-1, -1) if never flagged.
    """
    for idx, score in enumerate(tap_scores):
        if score < threshold:
            return idx, int(timesteps[idx])
    return -1, -1


def evaluate_detection(all_episodes, clean_episodes=None, success_based=True):
    """Compute detection metrics with clean-only calibration and trigger-style detection.

    Key design choices:
    - Threshold tau is calibrated on CLEAN SUCCESSES only (not pooled across conditions)
    - Trigger-style: episode is flagged if min(timestep_scores) < tau
    - Reports multiple operating points: TPR at 1%, 5%, 10% FPR

    Args:
        all_episodes: List of all episodes (clean + perturbed)
        clean_episodes: Clean episodes for threshold calibration. If None, uses
                        episodes with perturbation='clean'.
        success_based: If True, label by success/failure. If False, by clean/perturbed.

    Returns:
        Dict with detection metrics
    """
    # --- Identify calibration set: clean successes only ---
    if clean_episodes is None:
        clean_episodes = [ep for ep in all_episodes if ep['perturbation'] == 'clean']
    clean_success_episodes = [ep for ep in clean_episodes if ep['success']]

    # Calibration scores: min per-timestep score for each clean success episode
    # (trigger-style: the worst single timestep determines the episode)
    calib_min_scores = []
    for ep in clean_success_episodes:
        if len(ep.get('tap_scores', [])) > 0:
            calib_min_scores.append(float(np.min(ep['tap_scores'])))

    if success_based:
        # Label by actual episode outcome
        negative_episodes = [ep for ep in all_episodes if ep['success']]
        positive_episodes = [ep for ep in all_episodes if not ep['success']]
        label_type = 'success_vs_failure'
    else:
        # Label by clean vs perturbed
        negative_episodes = [ep for ep in all_episodes if ep['perturbation'] == 'clean']
        positive_episodes = [ep for ep in all_episodes if ep['perturbation'] != 'clean']
        label_type = 'clean_vs_perturbed'

    # Episode scores: use BOTH aggregation methods
    # - p10 for AUROC (smooth ranking)
    # - min for trigger-style detection (practical monitor)
    negative_p10 = [ep['tap_score'] for ep in negative_episodes]
    positive_p10 = [ep['tap_score'] for ep in positive_episodes]

    negative_min = [float(np.min(ep['tap_scores'])) if len(ep.get('tap_scores', [])) > 0 else 0.0
                    for ep in negative_episodes]
    positive_min = [float(np.min(ep['tap_scores'])) if len(ep.get('tap_scores', [])) > 0 else 0.0
                    for ep in positive_episodes]

    # --- AUROC using p10 scores (smooth, good for ranking) ---
    all_p10 = negative_p10 + positive_p10
    labels = [0] * len(negative_p10) + [1] * len(positive_p10)
    anomaly_scores = [-s for s in all_p10]

    if len(set(labels)) < 2:
        auroc_p10 = None
    else:
        auroc_p10 = roc_auc_score(labels, anomaly_scores)

    # --- AUROC using min scores (trigger-style) ---
    all_min = negative_min + positive_min
    anomaly_min = [-s for s in all_min]
    if len(set(labels)) < 2:
        auroc_min = None
    else:
        auroc_min = roc_auc_score(labels, anomaly_min)

    # Guard against empty splits
    if len(negative_p10) == 0 or len(positive_p10) == 0:
        return {
            'label_type': label_type, 'auroc_p10': None, 'auroc_min': None,
            'operating_points': {}, 'n_calib': 0, 'threshold': None,
            'n_negative': len(negative_p10), 'n_positive': len(positive_p10),
            'negative_mean_score': None, 'negative_std_score': None,
            'positive_mean_score': None, 'positive_std_score': None,
            'n_flagged_failures': 0, 'detection_indices': [], 'detection_timesteps': [],
            'mean_detection_index': None, 'median_detection_index': None,
            'mean_lead_time': None, 'median_lead_time': None,
            'pct_flagged_early': None,
        }

    # --- Calibrate thresholds on clean successes only ---
    # Use min-score distribution from clean successes for trigger-style thresholds
    operating_points = {}
    if len(calib_min_scores) >= 5:
        for fpr_target in [1, 5, 10]:
            tau = np.percentile(calib_min_scores, fpr_target)
            # TPR: fraction of positive episodes where min(scores) < tau
            tpr = float(np.mean([s < tau for s in positive_min]))
            # FPR on calibration set (clean successes) — should match target
            fpr_calib = float(np.mean([s < tau for s in calib_min_scores]))
            # FPR on all negatives (broader check)
            fpr_all_neg = float(np.mean([s < tau for s in negative_min]))
            operating_points[f'{fpr_target}%'] = {
                'tau': float(tau),
                'tpr': tpr,
                'fpr_calib': fpr_calib,
                'fpr_all_negative': fpr_all_neg,
            }
    else:
        # Fallback: calibrate on all negative min scores
        for fpr_target in [1, 5, 10]:
            tau = np.percentile(negative_min, fpr_target)
            tpr = float(np.mean([s < tau for s in positive_min]))
            fpr_all_neg = float(np.mean([s < tau for s in negative_min]))
            operating_points[f'{fpr_target}%'] = {
                'tau': float(tau),
                'tpr': tpr,
                'fpr_calib': None,
                'fpr_all_negative': fpr_all_neg,
            }

    # Use 5% FPR operating point as the primary threshold
    threshold = operating_points.get('5%', {}).get('tau', None)

    # --- Early warning: compute on POSITIVE (failure) episodes only ---
    # Detection time / lead time on successes would be false-alarm noise
    flagged_failures = []
    if threshold is not None:
        flagged_failures = [ep for ep in positive_episodes
                           if len(ep.get('tap_scores', [])) > 0
                           and float(np.min(ep['tap_scores'])) < threshold]

    detection_indices = []
    detection_timesteps = []
    lead_times = []
    early_flags = []
    for ep in flagged_failures:
        if 'tap_scores' in ep and 'tap_timesteps' in ep and len(ep['tap_timesteps']) > 0:
            idx, ts = first_flag(ep['tap_scores'], ep['tap_timesteps'], threshold)
            if idx >= 0:
                n_scored = len(ep['tap_timesteps'])
                detection_indices.append(idx)
                detection_timesteps.append(ts)
                lead_times.append((n_scored - 1) - idx)
                early_flags.append(idx < 0.7 * n_scored)

    return {
        'label_type': label_type,
        'auroc_p10': auroc_p10,
        'auroc_min': auroc_min,
        'operating_points': operating_points,
        'n_calib': len(calib_min_scores),
        'threshold': threshold,
        'n_negative': len(negative_p10),
        'n_positive': len(positive_p10),
        'negative_mean_score': float(np.mean(negative_p10)),
        'negative_std_score': float(np.std(negative_p10)),
        'positive_mean_score': float(np.mean(positive_p10)),
        'positive_std_score': float(np.std(positive_p10)),
        'n_flagged_failures': len(flagged_failures),
        'detection_indices': detection_indices,
        'detection_timesteps': detection_timesteps,
        'mean_detection_index': float(np.mean(detection_indices)) if detection_indices else None,
        'median_detection_index': float(np.median(detection_indices)) if detection_indices else None,
        'mean_lead_time': float(np.mean(lead_times)) if lead_times else None,
        'median_lead_time': float(np.median(lead_times)) if lead_times else None,
        'pct_flagged_early': float(np.mean(early_flags)) if early_flags else None,
    }


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy + TAP-Score Integration")
    parser.add_argument("--dp_checkpoint", type=str, required=True,
                        help="Path to Diffusion Policy checkpoint")
    parser.add_argument("--tap_checkpoint", type=str, required=True,
                        help="Path to TAP-Score checkpoint")
    parser.add_argument("--expert_data", type=str,
                        default="data/pusht/pusht_cchi_v7_replay.zarr",
                        help="Path to expert demonstrations zarr (for negative pool)")
    parser.add_argument("--n_clean_rollouts", type=int, default=50,
                        help="Number of clean rollouts")
    parser.add_argument("--n_perturbed_rollouts", type=int, default=50,
                        help="Number of perturbed rollouts")
    parser.add_argument("--perturbations", type=str, nargs='+',
                        default=["noise", "blur", "brightness", "contrast"],
                        choices=["noise", "blur", "brightness", "dark", "contrast", "occlusion"],
                        help="Types of perturbations to evaluate (space-separated)")
    parser.add_argument("--output_dir", type=str, default="eval_results/dp_with_tap",
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of parallel envs for batched DP inference (default: 10)")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Starting seed (same seeds used for clean and all perturbations)")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy=True for PushTImageEnv (may match published DP scores)")
    parser.add_argument("--success_threshold", type=float, default=0.80,
                        help="Max reward threshold for labeling episode as success (default: 0.80)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else
                                "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("Diffusion Policy + TAP-Score Integration")
    print("=" * 70)
    print(f"DP checkpoint: {args.dp_checkpoint}")
    print(f"TAP checkpoint: {args.tap_checkpoint}")
    print(f"Clean rollouts: {args.n_clean_rollouts}")
    print(f"Perturbed rollouts per type: {args.n_perturbed_rollouts}")
    print(f"Perturbations: {', '.join(args.perturbations)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed offset: {args.seed_offset}")
    print(f"Legacy env: {args.legacy}")
    print(f"Success threshold: {args.success_threshold}")
    print(f"Device: {args.device}")
    print("=" * 70)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Step 1: Load Models
    # ========================================================================
    print("\n[1/5] Loading models...")
    dp_policy, dp_cfg = load_diffusion_policy(args.dp_checkpoint, device)
    tap_model, tap_cfg = load_tap_score(args.tap_checkpoint, device)

    # Get action prediction parameters from DP config
    n_action_steps = dp_cfg.task.env_runner.n_action_steps
    n_obs_steps = dp_cfg.task.env_runner.n_obs_steps
    print(f"  DP action steps: {n_action_steps}, obs steps: {n_obs_steps}")

    # ========================================================================
    # Step 2: Build Negative Pool
    # ========================================================================
    print("\n[2/5] Building negative action pool...")
    negative_pool = build_negative_pool(
        args.expert_data,
        action_chunk=tap_cfg['action_chunk'],
        n_samples=2000
    )

    # ========================================================================
    # Step 3: Run Rollouts
    # ========================================================================
    print("\n[3/5] Running rollouts...")

    # Paired seeds: all conditions use the SAME seeds for fair comparison
    n_seeds = max(args.n_clean_rollouts, args.n_perturbed_rollouts)
    shared_seeds = list(range(args.seed_offset, args.seed_offset + n_seeds))
    clean_seeds = shared_seeds[:args.n_clean_rollouts]
    perturbed_seeds = shared_seeds[:args.n_perturbed_rollouts]

    print(f"  Paired seeds: {clean_seeds[0]}..{clean_seeds[-1]} (shared across all conditions)")

    all_episodes = []

    # Clean rollouts
    clean_episodes = run_rollouts(
        dp_policy,
        seeds=clean_seeds,
        perturbation_type=None,
        perturbation_name='clean',
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        device=device,
        batch_size=args.batch_size,
        legacy=args.legacy,
        success_threshold=args.success_threshold,
    )
    all_episodes.extend(clean_episodes)

    # Perturbed rollouts (multiple types) — same seeds as clean
    perturbed_by_type = {}
    for perturb_type in args.perturbations:
        perturbed_episodes = run_rollouts(
            dp_policy,
            seeds=perturbed_seeds,
            perturbation_type=perturb_type,
            perturbation_name=perturb_type,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            device=device,
            batch_size=args.batch_size,
            legacy=args.legacy,
            success_threshold=args.success_threshold,
        )
        perturbed_by_type[perturb_type] = perturbed_episodes
        all_episodes.extend(perturbed_episodes)

    # Report Diffusion Policy performance
    print(f"\nDiffusion Policy Performance:")
    clean_success_rate = np.mean([ep['success'] for ep in clean_episodes])
    print(f"  Clean: {clean_success_rate:.1%} success rate")

    for perturb_type, episodes in perturbed_by_type.items():
        success_rate = np.mean([ep['success'] for ep in episodes])
        print(f"  {perturb_type}: {success_rate:.1%} success rate")

    # ========================================================================
    # Step 4: Score Trajectories with TAP
    # ========================================================================
    print("\n[4/5] Scoring trajectories with TAP-Score...")

    for ep in tqdm(all_episodes, desc="TAP scoring"):
        scores, timesteps = score_trajectory_with_tap(tap_model, ep, negative_pool, tap_cfg, device)
        ep['tap_scores'] = scores
        ep['tap_timesteps'] = timesteps
        ep['tap_score'] = aggregate_episode_score(scores, method='percentile_10')

    # ========================================================================
    # Step 5: Compute Metrics
    # ========================================================================
    print("\n[5/5] Computing detection metrics...")

    # Two analyses: (1) success vs failure, (2) clean vs perturbed
    # Both calibrate thresholds on clean successes only
    metrics_success_based = evaluate_detection(all_episodes, clean_episodes=clean_episodes, success_based=True)
    metrics_perturbation_based = evaluate_detection(all_episodes, clean_episodes=clean_episodes, success_based=False)

    print(f"\nTAP-Score Detection Metrics (Success vs Failure):")
    print(f"  Calibration set: {metrics_success_based['n_calib']} clean success episodes")
    if metrics_success_based['auroc_p10'] is not None:
        print(f"  AUROC (p10 agg): {metrics_success_based['auroc_p10']:.3f}")
    if metrics_success_based['auroc_min'] is not None:
        print(f"  AUROC (min agg): {metrics_success_based['auroc_min']:.3f}")
    if metrics_success_based['negative_mean_score'] is not None:
        print(f"  Success mean score (p10): {metrics_success_based['negative_mean_score']:.3f} +/- {metrics_success_based['negative_std_score']:.3f}")
    if metrics_success_based['positive_mean_score'] is not None:
        print(f"  Failure mean score (p10): {metrics_success_based['positive_mean_score']:.3f} +/- {metrics_success_based['positive_std_score']:.3f}")

    if metrics_success_based['operating_points']:
        print(f"\n  Operating Points (trigger-style: flag if min(scores) < tau):")
        print(f"  {'FPR Target':>12}  {'tau':>8}  {'TPR':>8}  {'FPR(calib)':>12}  {'FPR(all neg)':>14}")
        for fpr_name, op in metrics_success_based['operating_points'].items():
            fpr_c = f"{op['fpr_calib']:>11.1%}" if op['fpr_calib'] is not None else f"{'N/A':>11}"
            print(f"  {fpr_name:>12}  {op['tau']:>8.3f}  {op['tpr']:>7.1%}  {fpr_c}  {op['fpr_all_negative']:>13.1%}")

    if metrics_success_based['mean_detection_index'] is not None:
        print(f"\n  Early Warning Metrics (failures only, n={metrics_success_based['n_flagged_failures']}):")
        print(f"    Mean detection index: {metrics_success_based['mean_detection_index']:.1f}")
        print(f"    Median detection index: {metrics_success_based['median_detection_index']:.1f}")
        print(f"    Mean lead time (steps remaining): {metrics_success_based['mean_lead_time']:.1f}")
        if metrics_success_based['pct_flagged_early'] is not None:
            print(f"    Flagged early (< 70% episode): {metrics_success_based['pct_flagged_early']:.1%}")

    print(f"\nTAP-Score Detection Metrics (Clean vs Perturbed):")
    if metrics_perturbation_based['auroc_p10'] is not None:
        print(f"  AUROC (p10): {metrics_perturbation_based['auroc_p10']:.3f}")
    if metrics_perturbation_based['auroc_min'] is not None:
        print(f"  AUROC (min): {metrics_perturbation_based['auroc_min']:.3f}")
    if metrics_perturbation_based['operating_points']:
        for fpr_name, op in metrics_perturbation_based['operating_points'].items():
            print(f"  TPR @ {fpr_name} FPR: {op['tpr']:.1%}")

    # ========================================================================
    # Save Results
    # ========================================================================

    # Compute per-perturbation statistics
    perturbation_stats = {}
    for perturb_type in ['clean'] + args.perturbations:
        episodes = [ep for ep in all_episodes if ep['perturbation'] == perturb_type]
        perturbation_stats[perturb_type] = {
            'n_episodes': len(episodes),
            'success_rate': float(np.mean([ep['success'] for ep in episodes])),
            'mean_reward': float(np.mean([ep['max_reward'] for ep in episodes])),
            'mean_tap_score': float(np.mean([ep['tap_score'] for ep in episodes])),
            'std_tap_score': float(np.std([ep['tap_score'] for ep in episodes])),
        }

    def _safe_float(v):
        """Convert to float, returning None if input is None."""
        return float(v) if v is not None else None

    results = {
        'config': {
            'dp_checkpoint': args.dp_checkpoint,
            'tap_checkpoint': args.tap_checkpoint,
            'n_clean_rollouts': args.n_clean_rollouts,
            'n_perturbed_rollouts': args.n_perturbed_rollouts,
            'perturbations': args.perturbations,
            'n_action_steps': int(n_action_steps),
            'n_obs_steps': int(n_obs_steps),
            'seed_offset': args.seed_offset,
            'paired_seeds': True,
            'legacy_env': args.legacy,
            'calibration': 'clean_successes_only',
            'detection_style': 'trigger_min',
            'success_threshold': args.success_threshold,
        },
        'perturbation_stats': perturbation_stats,
        'tap_score_detection': {
            'success_vs_failure': {
                'auroc_p10': _safe_float(metrics_success_based['auroc_p10']),
                'auroc_min': _safe_float(metrics_success_based['auroc_min']),
                'operating_points': metrics_success_based['operating_points'],
                'n_calib': metrics_success_based['n_calib'],
                'n_success': int(metrics_success_based['n_negative']),
                'n_failure': int(metrics_success_based['n_positive']),
                'success_mean_score': _safe_float(metrics_success_based['negative_mean_score']),
                'failure_mean_score': _safe_float(metrics_success_based['positive_mean_score']),
                'n_flagged_failures': metrics_success_based['n_flagged_failures'],
                'mean_detection_index': _safe_float(metrics_success_based['mean_detection_index']),
                'median_detection_index': _safe_float(metrics_success_based['median_detection_index']),
                'mean_lead_time': _safe_float(metrics_success_based['mean_lead_time']),
                'pct_flagged_early': _safe_float(metrics_success_based['pct_flagged_early']),
            },
            'clean_vs_perturbed': {
                'auroc_p10': _safe_float(metrics_perturbation_based['auroc_p10']),
                'auroc_min': _safe_float(metrics_perturbation_based['auroc_min']),
                'operating_points': metrics_perturbation_based['operating_points'],
                'n_clean': int(metrics_perturbation_based['n_negative']),
                'n_perturbed': int(metrics_perturbation_based['n_positive']),
                'clean_mean_score': _safe_float(metrics_perturbation_based['negative_mean_score']),
                'perturbed_mean_score': _safe_float(metrics_perturbation_based['positive_mean_score']),
            },
        },
        'episodes': [
            {
                'seed': int(ep['seed']),
                'perturbation': str(ep['perturbation']),
                'success': bool(ep['success']),
                'max_reward': float(ep['max_reward']),
                'tap_score_p10': float(ep['tap_score']),
                'tap_score_min': float(np.min(ep['tap_scores'])) if len(ep['tap_scores']) > 0 else None,
                'tap_score_mean': float(np.mean(ep['tap_scores'])) if len(ep['tap_scores']) > 0 else None,
                'tap_score_std': float(np.std(ep['tap_scores'])) if len(ep['tap_scores']) > 0 else None,
                'flag_index': first_flag(ep['tap_scores'], ep['tap_timesteps'],
                                        metrics_success_based['threshold'])[0]
                             if metrics_success_based['threshold'] is not None and len(ep['tap_scores']) > 0
                             else None,
            }
            for ep in all_episodes
        ]
    }

    output_path = output_dir / 'dp_with_tap_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, allow_nan=False)

    print(f"\n✓ Results saved to {output_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Paired seeds: {clean_seeds[0]}..{clean_seeds[-1]}")
    print(f"Diffusion Policy baseline: {clean_success_rate:.1%} clean success rate")
    for perturb_type, episodes in perturbed_by_type.items():
        success_rate = np.mean([ep['success'] for ep in episodes])
        print(f"  {perturb_type}: {success_rate:.1%} success rate")

    if metrics_success_based['auroc_min'] is not None:
        print(f"\nTAP-Score AUROC (min-trigger): {metrics_success_based['auroc_min']:.3f}")
        print(f"TAP-Score AUROC (p10):         {metrics_success_based['auroc_p10']:.3f}")

        op_5 = metrics_success_based['operating_points'].get('5%')
        if op_5:
            print(f"TPR @ 5% FPR (trigger): {op_5['tpr']:.1%} (tau={op_5['tau']:.3f})")

        if metrics_success_based['mean_lead_time'] is not None:
            print(f"Early warning: {metrics_success_based['n_flagged_failures']} failures flagged, "
                  f"lead time {metrics_success_based['mean_lead_time']:.1f} steps "
                  f"(median: {metrics_success_based['median_lead_time']:.1f})")

    print("=" * 70)


if __name__ == "__main__":
    main()
