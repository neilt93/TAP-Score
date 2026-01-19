"""
Robust TAP Evaluation with Held-Out Failure Types

Addresses reviewer concerns:
1. Held-out failure families (not in training negatives)
2. TPR at multiple FPR thresholds (1%, 5%, 10%)
3. Simple baselines (action magnitude, image difference)
4. Clean presentation of results

Training negatives: noise, permute, mirror, random
Held-out failures: scaling, bias, stuck, delayed (NOT in training)

Usage:
    python eval_tap_robust.py --checkpoint checkpoints_contrastive/contrastive_tap_best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

from tap.contrastive import ContrastiveTAPScore


M_NEGATIVES = 15


def load_model(checkpoint_path, device):
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
    return model, config


def load_episodes(data_path):
    root = zarr.open_group(str(data_path), mode='r')
    images = root['data']['img'][:]
    actions = root['data']['action'][:]
    episode_ends = root['meta']['episode_ends'][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episodes = []
    for i in range(len(episode_ends)):
        start, end = episode_starts[i], episode_ends[i]
        episodes.append({'images': images[start:end], 'actions': actions[start:end]})
    return episodes


def build_negative_pool(episodes, action_chunk, n_samples=2000):
    pool = []
    for _ in range(n_samples):
        ep = episodes[np.random.randint(len(episodes))]
        if len(ep['actions']) < action_chunk:
            continue
        t = np.random.randint(0, len(ep['actions']) - action_chunk)
        pool.append(ep['actions'][t:t + action_chunk].astype(np.float32))
    return np.stack(pool)


def perturb_observation(obs, severity=0.5):
    noise = np.random.randn(*obs.shape).astype(np.float32) * (0.3 * severity)
    return np.clip(obs + noise, 0, 1)


# ============================================================================
# FAILURE SIMULATIONS
# ============================================================================

def simulate_success_action(expert_action):
    """Success: small noise, stays on manifold."""
    noise = np.random.randn(*expert_action.shape).astype(np.float32) * 0.05
    return expert_action + noise


def simulate_failure_training_dist(expert_action):
    """
    Failure types that OVERLAP with training negatives.
    (Used for comparison, not main result)
    """
    failure_type = np.random.randint(3)
    if failure_type == 0:
        # Large noise (overlaps with training noise negative)
        noise = np.random.randn(*expert_action.shape).astype(np.float32) * 0.5
        return expert_action + noise
    elif failure_type == 1:
        # Temporal shift (overlaps with training permute negative)
        shift = np.random.randint(4, 10)
        return np.roll(expert_action, shift, axis=0)
    else:
        # Direction reversal (overlaps with training mirror negative)
        result = expert_action.copy()
        result[:, np.random.randint(expert_action.shape[1])] *= -1
        return result


def simulate_failure_heldout(expert_action):
    """
    HELD-OUT failure types NOT in training negatives.
    This is the key test for generalization.
    """
    failure_type = np.random.randint(4)
    if failure_type == 0:
        # ACTION SCALING: multiply by factor (NOT trained)
        scale = np.random.choice([0.5, 0.7, 1.5, 2.0])
        return (expert_action * scale).astype(np.float32)
    elif failure_type == 1:
        # CONSTANT BIAS: add offset (NOT trained)
        bias = np.random.randn(expert_action.shape[1]).astype(np.float32) * 0.3
        return (expert_action + bias).astype(np.float32)
    elif failure_type == 2:
        # STUCK POLICY: repeat first action (NOT trained)
        stuck = np.tile(expert_action[0:1], (expert_action.shape[0], 1))
        return stuck.astype(np.float32)
    else:
        # DELAYED REACTION: hold for k steps then apply (NOT trained)
        k = np.random.randint(3, 8)
        delayed = np.zeros_like(expert_action)
        delayed[:k] = expert_action[0]  # Hold initial action
        delayed[k:] = expert_action[:-k] if k < len(expert_action) else expert_action[0]
        return delayed.astype(np.float32)


# ============================================================================
# BASELINES
# ============================================================================

def baseline_action_magnitude(action_chunk):
    """
    Simple baseline: action norm.
    Higher norm often correlates with injected noise/scaling.
    """
    return np.linalg.norm(action_chunk)


def baseline_action_jerk(action_chunk):
    """
    Simple baseline: action jerk (second derivative).
    Higher jerk = less smooth = potentially off-manifold.
    """
    if len(action_chunk) < 3:
        return 0.0
    velocity = np.diff(action_chunk, axis=0)
    acceleration = np.diff(velocity, axis=0)
    return np.mean(np.abs(acceleration))


def baseline_image_difference(obs):
    """
    Simple baseline: mean image difference between frames.
    Doesn't use actions at all - pure visual metric.
    """
    if obs.shape[0] < 2:
        return 0.0
    diffs = np.abs(obs[1:] - obs[:-1])
    return np.mean(diffs)


# ============================================================================
# SCORING
# ============================================================================

def score_action_logmargin(model, obs, action, negative_pool, device):
    neg_indices = np.random.choice(len(negative_pool), M_NEGATIVES, replace=False)
    neg_actions = negative_pool[neg_indices]
    candidates = np.concatenate([action[np.newaxis], neg_actions], axis=0)

    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    candidates_t = torch.from_numpy(candidates).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(obs_t, candidates_t)
        l0 = logits[0, 0]
        l_neg = logits[0, 1:]
        margin = l0 - torch.logsumexp(l_neg, dim=0)

    return margin.item()


def score_episode(model, episode, negative_pool, device, config,
                  failure_fn=None, prefix_only=False):
    """Score episode, returning TAP margins and baseline scores."""
    obs_window = config.get("obs_window", 2)
    action_chunk = config.get("action_chunk", 16)
    images = episode['images']
    actions = episode['actions']
    ep_len = len(images)

    if ep_len < obs_window + action_chunk:
        return None

    end_t = ep_len - action_chunk
    if prefix_only:
        end_t = int((ep_len - action_chunk) * 0.7)

    tap_margins = []
    action_mags = []
    action_jerks = []
    img_diffs = []

    for t in range(obs_window - 1, end_t):
        start_t = t - obs_window + 1
        obs = images[start_t:t + 1].astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = perturb_observation(obs)  # Always perturb for consistency

        expert_action = actions[t:t + action_chunk].astype(np.float32)

        if failure_fn is None:
            policy_action = simulate_success_action(expert_action)
        else:
            policy_action = failure_fn(expert_action)

        # TAP score
        margin = score_action_logmargin(model, obs, policy_action, negative_pool, device)
        tap_margins.append(margin)

        # Baseline scores
        action_mags.append(baseline_action_magnitude(policy_action))
        action_jerks.append(baseline_action_jerk(policy_action))
        img_diffs.append(baseline_image_difference(obs))

    return {
        'tap_margins': np.array(tap_margins),
        'action_mags': np.array(action_mags),
        'action_jerks': np.array(action_jerks),
        'img_diffs': np.array(img_diffs),
    }


def aggregate_scores(scores_dict, method='percentile_10'):
    """Aggregate per-step scores to episode score."""
    result = {}
    for key, values in scores_dict.items():
        if len(values) == 0:
            result[key] = float('-inf') if 'tap' in key else 0.0
        elif method == 'percentile_10':
            result[key] = np.percentile(values, 10)
        else:
            result[key] = np.min(values)
    return result


def compute_tpr_at_fpr(scores, labels, target_fprs=[0.01, 0.05, 0.10]):
    """Compute TPR at multiple FPR thresholds."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    results = {}
    for target_fpr in target_fprs:
        # Find threshold closest to target FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        results[f'TPR@{int(target_fpr*100)}%FPR'] = tpr[idx]
        results[f'threshold@{int(target_fpr*100)}%FPR'] = thresholds[idx] if idx < len(thresholds) else None
    return results


def main():
    parser = argparse.ArgumentParser(description="Robust TAP Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_contrastive/contrastive_tap_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--n_episodes", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("Robust TAP Evaluation")
    print("=" * 70)
    print("Key test: HELD-OUT failure types (not in training negatives)")
    print("  - Action scaling (×0.5, ×0.7, ×1.5, ×2.0)")
    print("  - Constant bias (+offset)")
    print("  - Stuck policy (repeat action)")
    print("  - Delayed reaction (hold k steps)")
    print("=" * 70)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print("\nLoading...")
    model, config = load_model(args.checkpoint, device)
    episodes = load_episodes(args.data_dir)
    action_chunk = config.get("action_chunk", 16)
    negative_pool = build_negative_pool(episodes, action_chunk)

    np.random.seed(42)
    test_episodes = episodes[:args.n_episodes]

    results = {}

    # ========================================================================
    # 1. Success episodes (baseline)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Evaluating Success Episodes")
    print("=" * 70)

    success_tap = []
    success_mag = []
    success_jerk = []

    for ep in tqdm(test_episodes, desc="Success"):
        scores = score_episode(model, ep, negative_pool, device, config, failure_fn=None)
        if scores is not None:
            agg = aggregate_scores(scores)
            success_tap.append(agg['tap_margins'])
            success_mag.append(agg['action_mags'])
            success_jerk.append(agg['action_jerks'])

    print(f"Success TAP mean: {np.mean(success_tap):.2f}")

    # ========================================================================
    # 2. Training-distribution failures (for comparison)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Evaluating Training-Dist Failures (overlaps with training)")
    print("=" * 70)

    train_fail_tap = []
    train_fail_mag = []
    train_fail_jerk = []

    for ep in tqdm(test_episodes, desc="Train-dist failure"):
        scores = score_episode(model, ep, negative_pool, device, config,
                               failure_fn=simulate_failure_training_dist)
        if scores is not None:
            agg = aggregate_scores(scores)
            train_fail_tap.append(agg['tap_margins'])
            train_fail_mag.append(agg['action_mags'])
            train_fail_jerk.append(agg['action_jerks'])

    print(f"Train-dist failure TAP mean: {np.mean(train_fail_tap):.2f}")

    # AUROC for training-dist
    all_tap = success_tap + train_fail_tap
    labels = [0] * len(success_tap) + [1] * len(train_fail_tap)
    train_auroc = roc_auc_score(labels, [-s for s in all_tap])
    print(f"Training-dist Failure AUROC: {train_auroc:.3f}")

    # ========================================================================
    # 3. HELD-OUT failures (KEY TEST)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Evaluating HELD-OUT Failures (NOT in training)")
    print("=" * 70)

    heldout_fail_tap = []
    heldout_fail_mag = []
    heldout_fail_jerk = []

    for ep in tqdm(test_episodes, desc="Held-out failure"):
        scores = score_episode(model, ep, negative_pool, device, config,
                               failure_fn=simulate_failure_heldout)
        if scores is not None:
            agg = aggregate_scores(scores)
            heldout_fail_tap.append(agg['tap_margins'])
            heldout_fail_mag.append(agg['action_mags'])
            heldout_fail_jerk.append(agg['action_jerks'])

    print(f"Held-out failure TAP mean: {np.mean(heldout_fail_tap):.2f}")

    # AUROC for held-out
    all_tap_heldout = success_tap + heldout_fail_tap
    labels_heldout = [0] * len(success_tap) + [1] * len(heldout_fail_tap)
    heldout_auroc = roc_auc_score(labels_heldout, [-s for s in all_tap_heldout])
    print(f"HELD-OUT Failure AUROC: {heldout_auroc:.3f}")

    # TPR at FPR thresholds
    tpr_results = compute_tpr_at_fpr([-s for s in all_tap_heldout], labels_heldout)
    print("\nTPR at FPR thresholds (held-out failures):")
    for k, v in tpr_results.items():
        if 'TPR' in k:
            print(f"  {k}: {v:.1%}")

    # ========================================================================
    # 4. Baseline comparisons
    # ========================================================================
    print("\n" + "=" * 70)
    print("Baseline Comparisons")
    print("=" * 70)

    # Action magnitude baseline
    all_mag = success_mag + heldout_fail_mag
    mag_auroc = roc_auc_score(labels_heldout, all_mag)  # Higher mag = more anomalous
    print(f"Action Magnitude AUROC: {mag_auroc:.3f}")

    # Action jerk baseline
    all_jerk = success_jerk + heldout_fail_jerk
    jerk_auroc = roc_auc_score(labels_heldout, all_jerk)
    print(f"Action Jerk AUROC: {jerk_auroc:.3f}")

    print(f"\nTAP improvement over best baseline: {(heldout_auroc - max(mag_auroc, jerk_auroc)) * 100:.1f}%")

    # ========================================================================
    # 5. Prefix-only (early warning)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Prefix-Only Evaluation (Early Warning)")
    print("=" * 70)

    prefix_success = []
    prefix_heldout = []

    for ep in tqdm(test_episodes[:20], desc="Prefix success"):
        scores = score_episode(model, ep, negative_pool, device, config,
                               failure_fn=None, prefix_only=True)
        if scores is not None and len(scores['tap_margins']) > 0:
            prefix_success.append(np.percentile(scores['tap_margins'], 10))

    for ep in tqdm(test_episodes[:20], desc="Prefix held-out"):
        scores = score_episode(model, ep, negative_pool, device, config,
                               failure_fn=simulate_failure_heldout, prefix_only=True)
        if scores is not None and len(scores['tap_margins']) > 0:
            prefix_heldout.append(np.percentile(scores['tap_margins'], 10))

    if len(prefix_success) > 0 and len(prefix_heldout) > 0:
        prefix_all = prefix_success + prefix_heldout
        prefix_labels = [0] * len(prefix_success) + [1] * len(prefix_heldout)
        prefix_auroc = roc_auc_score(prefix_labels, [-s for s in prefix_all])
        print(f"Prefix-Only AUROC (held-out): {prefix_auroc:.3f}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print("\n| Metric                          | Value  |")
    print("|--------------------------------|--------|")
    print(f"| Training-dist Failure AUROC    | {train_auroc:.3f}  |")
    print(f"| **HELD-OUT Failure AUROC**     | **{heldout_auroc:.3f}**  |")
    print(f"| Prefix-Only AUROC (held-out)   | {prefix_auroc:.3f}  |" if 'prefix_auroc' in dir() else "")
    print(f"| Action Magnitude Baseline      | {mag_auroc:.3f}  |")
    print(f"| Action Jerk Baseline           | {jerk_auroc:.3f}  |")

    print("\n| FPR   | TPR (held-out) |")
    print("|-------|----------------|")
    for k, v in tpr_results.items():
        if 'TPR' in k:
            fpr_str = k.replace('TPR@', '').replace('FPR', '')
            print(f"| {fpr_str:5} | {v:.1%}          |")

    print(f"\nScore separation:")
    print(f"  Success mean:        {np.mean(success_tap):.2f}")
    print(f"  Train-dist fail:     {np.mean(train_fail_tap):.2f}")
    print(f"  Held-out fail:       {np.mean(heldout_fail_tap):.2f}")

    # Save results
    results = {
        'train_dist_auroc': float(train_auroc),
        'heldout_auroc': float(heldout_auroc),
        'prefix_auroc': float(prefix_auroc) if 'prefix_auroc' in dir() else None,
        'baseline_magnitude_auroc': float(mag_auroc),
        'baseline_jerk_auroc': float(jerk_auroc),
        'tpr_at_fpr': {k: float(v) if v is not None else None for k, v in tpr_results.items()},
        'success_mean': float(np.mean(success_tap)),
        'train_fail_mean': float(np.mean(train_fail_tap)),
        'heldout_fail_mean': float(np.mean(heldout_fail_tap)),
    }

    with open(output_dir / "robust_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'robust_eval_results.json'}")


if __name__ == "__main__":
    main()
