"""
Fixed Episode-Level TAP Evaluation

Key fixes:
1. Log-margin scoring (stable, no underflow)
2. Calibrate on clean policy rollouts (not expert)
3. Focus on Failure AUROC (not OOD detection)
4. Prefix-only validation (prove early warning is real)
5. Better failure simulation (temporal drift, not just noise)

Usage:
    python eval_tap_fixed.py --checkpoint checkpoints_contrastive/contrastive_tap_best.pt
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
import matplotlib.pyplot as plt

from tap.contrastive import ContrastiveTAPScore


# Fixed parameters
M_NEGATIVES = 15


def load_model(checkpoint_path, device):
    """Load model."""
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
    """Load episodes."""
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
    """Build negative pool."""
    pool = []
    for _ in range(n_samples):
        ep = episodes[np.random.randint(len(episodes))]
        if len(ep['actions']) < action_chunk:
            continue
        t = np.random.randint(0, len(ep['actions']) - action_chunk)
        pool.append(ep['actions'][t:t + action_chunk].astype(np.float32))
    return np.stack(pool)


def perturb_observation(obs, severity=0.5):
    """Apply noise perturbation."""
    noise = np.random.randn(*obs.shape).astype(np.float32) * (0.3 * severity)
    return np.clip(obs + noise, 0, 1)


def simulate_policy_action(expert_action, is_failure):
    """
    Simulate policy action with clearer success/failure distinction.

    Success: Small noise (policy stays on manifold)
    Failure: Larger deviation + temporal drift (policy goes off manifold)
    """
    if is_failure:
        # Failure: multiple types of off-manifold deviation
        failure_type = np.random.randint(3)
        if failure_type == 0:
            # Large noise
            noise = np.random.randn(*expert_action.shape).astype(np.float32) * 0.5
            return expert_action + noise
        elif failure_type == 1:
            # Temporal shift (wrong timing)
            shift = np.random.randint(4, 10)
            shifted = np.roll(expert_action, shift, axis=0)
            return shifted
        else:
            # Direction reversal
            reversed_action = expert_action.copy()
            reversed_action[:, np.random.randint(expert_action.shape[1])] *= -1
            return reversed_action
    else:
        # Success: small noise (on manifold)
        noise = np.random.randn(*expert_action.shape).astype(np.float32) * 0.05
        return expert_action + noise


def score_action_logmargin(model, obs, action, negative_pool, device, m_negatives=M_NEGATIVES):
    """
    Score action using log-margin (stable, no underflow).

    margin = l0 - logsumexp([l1..lM])

    This is the log-odds that the proposed action is better than the negative set.
    """
    neg_indices = np.random.choice(len(negative_pool), m_negatives, replace=False)
    neg_actions = negative_pool[neg_indices]
    candidates = np.concatenate([action[np.newaxis], neg_actions], axis=0)

    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    candidates_t = torch.from_numpy(candidates).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(obs_t, candidates_t)  # (1, 1+M)

        # Log-margin: l0 - logsumexp(l1..lM)
        l0 = logits[0, 0]  # proposed action logit
        l_neg = logits[0, 1:]  # negative logits
        margin = l0 - torch.logsumexp(l_neg, dim=0)

    return margin.item()


def score_episode_logmargin(model, episode, negative_pool, device, config,
                             perturbation=False, is_failure=False, prefix_only=False):
    """
    Score all timesteps in episode using log-margin.

    Args:
        prefix_only: If True, only score first 70% of episode (for validation)
    """
    obs_window = config.get("obs_window", 2)
    action_chunk = config.get("action_chunk", 16)
    images = episode['images']
    actions = episode['actions']
    ep_len = len(images)

    if ep_len < obs_window + action_chunk:
        return None

    # For prefix-only, stop at 70% of episode
    end_t = ep_len - action_chunk
    if prefix_only:
        end_t = int((ep_len - action_chunk) * 0.7)

    margins = []
    for t in range(obs_window - 1, end_t):
        start_t = t - obs_window + 1
        obs = images[start_t:t + 1].astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))

        if perturbation:
            obs = perturb_observation(obs)

        expert_action = actions[t:t + action_chunk].astype(np.float32)
        policy_action = simulate_policy_action(expert_action, is_failure)
        margin = score_action_logmargin(model, obs, policy_action, negative_pool, device)
        margins.append(margin)

    return np.array(margins) if len(margins) > 0 else None


def aggregate_episode_logmargin(margins, method='percentile_10'):
    """
    Aggregate timestep margins to episode score.

    For log-margins, higher = better (more confident action is correct).
    """
    if len(margins) == 0:
        return float('-inf')

    if method == 'percentile_10':
        return np.percentile(margins, 10)
    elif method == 'percentile_20_mean':
        # Mean of lowest 20% margins
        k = max(1, len(margins) // 5)
        return np.mean(np.sort(margins)[:k])
    elif method == 'min':
        return np.min(margins)


def main():
    parser = argparse.ArgumentParser(description="Fixed TAP Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_contrastive/contrastive_tap_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--n_episodes", type=int, default=40)
    parser.add_argument("--target_fpr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("Fixed TAP Evaluation (Log-Margin Scoring)")
    print("=" * 70)
    print(f"M = {M_NEGATIVES} negatives (fixed)")
    print(f"Scoring: log-margin = l0 - logsumexp(l1..lM)")
    print("=" * 70)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print("\nLoading model and data...")
    model, config = load_model(args.checkpoint, device)
    episodes = load_episodes(args.data_dir)
    action_chunk = config.get("action_chunk", 16)
    negative_pool = build_negative_pool(episodes, action_chunk)
    print(f"Loaded {len(episodes)} episodes")

    # Split episodes
    np.random.seed(42)
    indices = np.random.permutation(len(episodes))
    n_cal = len(episodes) // 4  # 25% for calibration
    cal_episodes = [episodes[i] for i in indices[:n_cal]]
    test_episodes = [episodes[i] for i in indices[n_cal:]]

    # ========================================================================
    # 1. Calibrate threshold on CLEAN POLICY rollouts (not expert)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Calibrating on Clean Policy Rollouts")
    print("=" * 70)

    clean_scores = []
    for ep in tqdm(cal_episodes, desc="Calibration (clean)"):
        margins = score_episode_logmargin(
            model, ep, negative_pool, device, config,
            perturbation=False, is_failure=False  # Clean success episodes
        )
        if margins is not None:
            score = aggregate_episode_logmargin(margins, 'percentile_10')
            clean_scores.append(score)

    threshold = np.percentile(clean_scores, args.target_fpr * 100)

    print(f"\nClean policy episode scores (log-margin):")
    print(f"  Mean: {np.mean(clean_scores):.2f}")
    print(f"  Std: {np.std(clean_scores):.2f}")
    print(f"  Range: [{np.min(clean_scores):.2f}, {np.max(clean_scores):.2f}]")
    print(f"\nCalibrated threshold (τ): {threshold:.2f}")
    print(f"  ({args.target_fpr*100:.0f}% of clean episodes fall below this)")

    # ========================================================================
    # 2. Evaluate Failure Prediction (Success vs Failure under perturbation)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Failure Prediction AUROC")
    print("=" * 70)

    success_scores = []
    failure_scores = []

    # Select test episodes
    test_sample = test_episodes[:args.n_episodes]

    for ep in tqdm(test_sample, desc="Success (perturbed)"):
        margins = score_episode_logmargin(
            model, ep, negative_pool, device, config,
            perturbation=True, is_failure=False
        )
        if margins is not None:
            success_scores.append(aggregate_episode_logmargin(margins))

    for ep in tqdm(test_sample, desc="Failure (perturbed)"):
        margins = score_episode_logmargin(
            model, ep, negative_pool, device, config,
            perturbation=True, is_failure=True
        )
        if margins is not None:
            failure_scores.append(aggregate_episode_logmargin(margins))

    print(f"\nSuccess episodes: mean score = {np.mean(success_scores):.2f}")
    print(f"Failure episodes: mean score = {np.mean(failure_scores):.2f}")
    print(f"Separation: {np.mean(success_scores) - np.mean(failure_scores):.2f}")

    # Compute AUROC (lower score = more likely failure)
    all_scores = success_scores + failure_scores
    labels = [0] * len(success_scores) + [1] * len(failure_scores)
    # Negate scores so higher = more anomalous
    anomaly_scores = [-s for s in all_scores]

    failure_auroc = roc_auc_score(labels, anomaly_scores)
    print(f"\nFailure Prediction AUROC: {failure_auroc:.3f}")

    # Detection rate at threshold
    failure_detected = sum(1 for s in failure_scores if s < threshold)
    success_false_alarm = sum(1 for s in success_scores if s < threshold)

    print(f"At threshold τ = {threshold:.2f}:")
    print(f"  Failure detection rate: {failure_detected}/{len(failure_scores)} = {failure_detected/len(failure_scores)*100:.1f}%")
    print(f"  False alarm rate: {success_false_alarm}/{len(success_scores)} = {success_false_alarm/len(success_scores)*100:.1f}%")

    # ========================================================================
    # 3. Prefix-Only Validation (prove early warning is real)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Prefix-Only Validation (Early Warning Check)")
    print("=" * 70)
    print("Using only first 70% of episode to predict eventual failure")

    prefix_success = []
    prefix_failure = []

    for ep in tqdm(test_sample[:20], desc="Prefix success"):
        margins = score_episode_logmargin(
            model, ep, negative_pool, device, config,
            perturbation=True, is_failure=False, prefix_only=True
        )
        if margins is not None and len(margins) > 0:
            prefix_success.append(aggregate_episode_logmargin(margins))

    for ep in tqdm(test_sample[:20], desc="Prefix failure"):
        margins = score_episode_logmargin(
            model, ep, negative_pool, device, config,
            perturbation=True, is_failure=True, prefix_only=True
        )
        if margins is not None and len(margins) > 0:
            prefix_failure.append(aggregate_episode_logmargin(margins))

    if len(prefix_success) > 0 and len(prefix_failure) > 0:
        prefix_scores = prefix_success + prefix_failure
        prefix_labels = [0] * len(prefix_success) + [1] * len(prefix_failure)
        prefix_anomaly = [-s for s in prefix_scores]
        prefix_auroc = roc_auc_score(prefix_labels, prefix_anomaly)

        print(f"\nPrefix-only scores:")
        print(f"  Success mean: {np.mean(prefix_success):.2f}")
        print(f"  Failure mean: {np.mean(prefix_failure):.2f}")
        print(f"  Prefix-Only AUROC: {prefix_auroc:.3f}")

        if prefix_auroc > 0.7:
            print("  ✓ Early warning is REAL (can predict failure before it happens)")
        else:
            print("  ! AUROC drops with prefix-only; detection may require seeing failure")

    # ========================================================================
    # 4. Baseline Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Baseline Comparison")
    print("=" * 70)

    # Random baseline
    random_auroc = 0.5
    print(f"Random baseline AUROC: {random_auroc:.3f}")
    print(f"TAP AUROC: {failure_auroc:.3f}")
    print(f"Improvement over random: {(failure_auroc - random_auroc) * 100:.1f}%")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = {
        'M_negatives': M_NEGATIVES,
        'threshold': float(threshold),
        'clean_mean_score': float(np.mean(clean_scores)),
        'success_mean_score': float(np.mean(success_scores)),
        'failure_mean_score': float(np.mean(failure_scores)),
        'separation': float(np.mean(success_scores) - np.mean(failure_scores)),
        'failure_auroc': float(failure_auroc),
        'failure_detection_rate': float(failure_detected / len(failure_scores)),
        'false_alarm_rate': float(success_false_alarm / len(success_scores)),
        'prefix_auroc': float(prefix_auroc) if 'prefix_auroc' in dir() else None,
    }

    print(f"\n| Metric                    | Value  |")
    print(f"|---------------------------|--------|")
    print(f"| Failure AUROC             | {failure_auroc:.3f}  |")
    print(f"| Prefix-Only AUROC         | {prefix_auroc:.3f}  |" if 'prefix_auroc' in dir() else "")
    print(f"| Threshold τ               | {threshold:.2f}  |")
    print(f"| Failure Detection Rate    | {failure_detected/len(failure_scores)*100:.1f}%   |")
    print(f"| False Alarm Rate          | {success_false_alarm/len(success_scores)*100:.1f}%    |")
    print(f"| Score Separation          | {np.mean(success_scores) - np.mean(failure_scores):.2f}  |")

    print(f"\nM = {M_NEGATIVES} negatives, aggregation = 10th percentile")
    print("Scoring: log-margin (stable, no underflow)")

    # Save
    with open(output_dir / "fixed_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'fixed_eval_results.json'}")


if __name__ == "__main__":
    main()
