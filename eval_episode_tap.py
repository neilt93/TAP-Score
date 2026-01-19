"""
Episode-Level Contrastive TAP Evaluation

Computes episode-level metrics that prove practical value:
- OOD AUROC: clean vs perturbed episodes
- Failure AUROC: success vs failure under perturbation
- Early warning lead time
- Threshold calibration on clean validation

Usage:
    python eval_episode_tap.py --checkpoint checkpoints_contrastive/contrastive_tap_best.pt
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import zarr
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from tap.contrastive import ContrastiveTAPScore


# ============================================================================
# Constants - Keep fixed for all evaluations
# ============================================================================
M_NEGATIVES = 15  # Number of negative samples for scoring (keep fixed!)
TEMPERATURE = 0.1  # Model temperature (keep fixed!)


def load_model(checkpoint_path, device):
    """Load contrastive TAP model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = ContrastiveTAPScore(
        obs_channels=config.get("obs_channels", 3),
        action_dim=config.get("action_dim", 2),
        obs_window=config.get("obs_window", 2),
        action_chunk=config.get("action_chunk", 16),
        hidden_dim=config.get("hidden_dim", 128),
        temperature=config.get("temperature", TEMPERATURE),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def load_episodes(data_path):
    """Load episodes from zarr dataset."""
    root = zarr.open_group(str(data_path), mode='r')
    images = root['data']['img'][:]
    actions = root['data']['action'][:]
    episode_ends = root['meta']['episode_ends'][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    episodes = []
    for i in range(len(episode_ends)):
        start, end = episode_starts[i], episode_ends[i]
        episodes.append({
            'images': images[start:end],
            'actions': actions[start:end],
            'episode_idx': i,
        })

    return episodes


def build_negative_pool(episodes, action_chunk, n_samples=2000):
    """Build fixed pool of negative actions for scoring consistency."""
    pool = []
    for _ in range(n_samples):
        ep = episodes[np.random.randint(len(episodes))]
        if len(ep['actions']) < action_chunk:
            continue
        t = np.random.randint(0, len(ep['actions']) - action_chunk)
        pool.append(ep['actions'][t:t + action_chunk].astype(np.float32))

    return np.stack(pool)


def perturb_observation(obs, perturbation, severity=0.5):
    """Apply perturbation to observation."""
    obs = obs.copy()

    if perturbation == "noise":
        noise = np.random.randn(*obs.shape).astype(np.float32) * (0.3 * severity)
        obs = np.clip(obs + noise, 0, 1)
    elif perturbation == "blur":
        from scipy.ndimage import gaussian_filter
        for t in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                obs[t, c] = gaussian_filter(obs[t, c], sigma=2.0 * severity)
    elif perturbation == "brightness":
        factor = 1.0 + (0.5 * severity)
        obs = np.clip(obs * factor, 0, 1)
    elif perturbation == "contrast":
        mean = obs.mean()
        obs = np.clip((obs - mean) * (1.0 + 0.5 * severity) + mean, 0, 1)

    return obs


def simulate_policy_action(expert_action, is_failure, severity=0.3):
    """
    Simulate what a policy might produce.

    For success: small deviation from expert
    For failure: larger deviation (off-manifold)
    """
    if is_failure:
        # Failed policy: larger noise or temporal shift
        if np.random.rand() > 0.5:
            # Add significant noise
            noise = np.random.randn(*expert_action.shape).astype(np.float32) * severity * 2
            return expert_action + noise
        else:
            # Temporal misalignment (shift actions)
            shift = np.random.randint(2, 6)
            shifted = np.roll(expert_action, shift, axis=0)
            return shifted
    else:
        # Successful policy: small noise
        noise = np.random.randn(*expert_action.shape).astype(np.float32) * severity * 0.3
        return expert_action + noise


def score_action(model, obs, action, negative_pool, device, m_negatives=M_NEGATIVES):
    """
    Score (obs, action) pair using contrastive TAP.

    Returns softmax probability that action is best among M+1 candidates.
    M is kept fixed for calibration consistency.
    """
    # Sample negatives
    neg_indices = np.random.choice(len(negative_pool), m_negatives, replace=False)
    neg_actions = negative_pool[neg_indices]

    # Construct candidates: [proposal, neg1, neg2, ...]
    candidates = np.concatenate([action[np.newaxis], neg_actions], axis=0)

    # Convert to tensors
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    candidates_t = torch.from_numpy(candidates).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(obs_t, candidates_t)
        probs = torch.softmax(logits, dim=-1)
        score = probs[0, 0].item()

    return score


def score_episode(model, episode, negative_pool, device, obs_window, action_chunk,
                  perturbation=None, is_failure=False, score_expert=False):
    """
    Score all timesteps in an episode.

    Returns:
        policy_scores: TAP scores for simulated policy actions
        expert_scores: TAP scores for expert actions (if score_expert=True)
        timesteps: valid timestep indices
    """
    images = episode['images']
    actions = episode['actions']
    ep_len = len(images)

    if ep_len < obs_window + action_chunk:
        return None, None, None

    policy_scores = []
    expert_scores = []
    timesteps = []

    for t in range(obs_window - 1, ep_len - action_chunk):
        # Get observation window
        start_t = t - obs_window + 1
        obs = images[start_t:t + 1].astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))  # (T, C, H, W)

        # Apply perturbation if specified
        if perturbation:
            obs = perturb_observation(obs, perturbation)

        # Get expert action chunk
        expert_action = actions[t:t + action_chunk].astype(np.float32)

        # Simulate policy action
        policy_action = simulate_policy_action(expert_action, is_failure)

        # Score policy action
        policy_score = score_action(model, obs, policy_action, negative_pool, device)
        policy_scores.append(policy_score)

        # Optionally score expert action
        if score_expert:
            expert_score = score_action(model, obs, expert_action, negative_pool, device)
            expert_scores.append(expert_score)

        timesteps.append(t)

    return np.array(policy_scores), np.array(expert_scores) if score_expert else None, timesteps


def aggregate_episode_score(scores, method='percentile_10'):
    """
    Aggregate timestep scores to episode score.

    Methods:
        'min': minimum score (most sensitive to outliers)
        'percentile_10': 10th percentile (more robust)
        'percentile_5': 5th percentile
        'mean': average (least sensitive)
    """
    if len(scores) == 0:
        return 0.0

    if method == 'min':
        return np.min(scores)
    elif method == 'percentile_10':
        return np.percentile(scores, 10)
    elif method == 'percentile_5':
        return np.percentile(scores, 5)
    elif method == 'mean':
        return np.mean(scores)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_early_warning_lead_time(scores, threshold, timesteps, action_chunk):
    """
    Compute early warning lead time for failed episodes.

    Returns: number of steps between first threshold crossing and episode end
    """
    for i, score in enumerate(scores):
        if score < threshold:
            # First crossing - compute lead time
            steps_remaining = len(scores) - i
            return steps_remaining * action_chunk  # Convert to actual steps

    return 0  # Never crossed threshold


def run_episode_evaluation(model, episodes, negative_pool, device, config,
                           perturbation=None, n_episodes=50, aggregation='percentile_10'):
    """
    Run episode-level evaluation.

    Returns dict with:
        - episode_scores: list of (score, is_failure, is_perturbed) tuples
        - policy_trajectories: for plotting
        - expert_trajectories: for plotting
    """
    obs_window = config.get("obs_window", 2)
    action_chunk = config.get("action_chunk", 16)

    results = {
        'scores': [],
        'trajectories': [],
    }

    # Sample episodes
    selected = np.random.choice(len(episodes), min(n_episodes, len(episodes)), replace=False)

    for idx in tqdm(selected, desc=f"Episodes ({perturbation or 'clean'})"):
        episode = episodes[idx]

        # Randomly assign success/failure (50/50 for simulation)
        is_failure = np.random.rand() > 0.5

        # Score episode
        policy_scores, expert_scores, timesteps = score_episode(
            model, episode, negative_pool, device,
            obs_window, action_chunk,
            perturbation=perturbation,
            is_failure=is_failure,
            score_expert=True
        )

        if policy_scores is None:
            continue

        # Aggregate to episode score
        episode_score = aggregate_episode_score(policy_scores, aggregation)

        results['scores'].append({
            'score': episode_score,
            'is_failure': is_failure,
            'is_perturbed': perturbation is not None,
            'perturbation': perturbation,
            'episode_idx': int(episode['episode_idx']),
        })

        results['trajectories'].append({
            'policy_scores': policy_scores.tolist(),
            'expert_scores': expert_scores.tolist() if expert_scores is not None else None,
            'timesteps': timesteps,
            'is_failure': is_failure,
            'is_perturbed': perturbation is not None,
            'perturbation': perturbation,
        })

    return results


def compute_auroc(scores, labels):
    """Compute AUROC, handling edge cases."""
    if len(set(labels)) < 2:
        return None  # Can't compute AUROC with single class
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return None


def calibrate_threshold(clean_scores, target_fpr=0.05):
    """
    Calibrate threshold on clean validation episodes.

    Returns threshold tau such that FPR = target_fpr on clean data.
    (i.e., tau = (1 - target_fpr) percentile of clean scores)
    """
    # For TAP scores, higher is better (more compatible)
    # So threshold = 5th percentile means 5% of clean episodes fall below
    tau = np.percentile(clean_scores, target_fpr * 100)
    return tau


def compute_detection_rates(scores, labels, threshold):
    """
    Compute detection rates at fixed threshold.

    Args:
        scores: episode scores (higher = more compatible = less anomalous)
        labels: 1 for positive (perturbed/failed), 0 for negative (clean/success)
        threshold: score below which we flag as detected

    Returns:
        tpr: true positive rate (detection rate)
        fpr: false positive rate (false alarm rate)
    """
    predictions = (np.array(scores) < threshold).astype(int)
    labels = np.array(labels)

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return tpr, fpr


def plot_episode_trajectories(trajectories, output_path, n_examples=4):
    """
    Plot TAP score over time for success vs failure episodes.

    Shows the key claim: policy scores drop more than expert under shift.
    """
    # Separate by condition
    success_perturbed = [t for t in trajectories if not t['is_failure'] and t['is_perturbed']]
    failure_perturbed = [t for t in trajectories if t['is_failure'] and t['is_perturbed']]

    if len(success_perturbed) == 0 or len(failure_perturbed) == 0:
        print("Not enough trajectories for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot success example
    if len(success_perturbed) > 0:
        traj = success_perturbed[0]
        ax = axes[0, 0]
        ax.plot(traj['policy_scores'], label='Policy action', color='blue', linewidth=2)
        if traj['expert_scores']:
            ax.plot(traj['expert_scores'], label='Expert action', color='green', linewidth=2, linestyle='--')
        ax.axhline(y=1/(M_NEGATIVES+1), color='gray', linestyle=':', label=f'Random baseline (1/{M_NEGATIVES+1})')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('TAP Score')
        ax.set_title(f"Success Episode (perturbed: {traj['perturbation']})")
        ax.legend()
        ax.set_ylim(0, 1)

    # Plot failure example
    if len(failure_perturbed) > 0:
        traj = failure_perturbed[0]
        ax = axes[0, 1]
        ax.plot(traj['policy_scores'], label='Policy action', color='red', linewidth=2)
        if traj['expert_scores']:
            ax.plot(traj['expert_scores'], label='Expert action', color='green', linewidth=2, linestyle='--')
        ax.axhline(y=1/(M_NEGATIVES+1), color='gray', linestyle=':', label=f'Random baseline')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('TAP Score')
        ax.set_title(f"Failure Episode (perturbed: {traj['perturbation']})")
        ax.legend()
        ax.set_ylim(0, 1)

    # Score distributions
    ax = axes[1, 0]
    success_scores = [aggregate_episode_score(np.array(t['policy_scores'])) for t in success_perturbed]
    failure_scores = [aggregate_episode_score(np.array(t['policy_scores'])) for t in failure_perturbed]
    ax.hist(success_scores, bins=20, alpha=0.5, label='Success', color='blue')
    ax.hist(failure_scores, bins=20, alpha=0.5, label='Failure', color='red')
    ax.set_xlabel('Episode TAP Score (10th percentile)')
    ax.set_ylabel('Count')
    ax.set_title('Episode Score Distribution')
    ax.legend()

    # Expert vs Policy gap under perturbation
    ax = axes[1, 1]
    policy_means = []
    expert_means = []
    for traj in trajectories:
        if traj['is_perturbed'] and traj['expert_scores']:
            policy_means.append(np.mean(traj['policy_scores']))
            expert_means.append(np.mean(traj['expert_scores']))

    if len(policy_means) > 0:
        ax.scatter(expert_means, policy_means, alpha=0.5)
        ax.plot([0, 1], [0, 1], 'k--', label='y=x')
        ax.set_xlabel('Expert Action Score (mean)')
        ax.set_ylabel('Policy Action Score (mean)')
        ax.set_title('Expert vs Policy Scores Under Perturbation')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Episode-Level TAP Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_contrastive/contrastive_tap_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--aggregation", type=str, default="percentile_10",
                        choices=["min", "percentile_10", "percentile_5", "mean"])
    parser.add_argument("--target_fpr", type=float, default=0.05, help="Target false positive rate for threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("Episode-Level Contrastive TAP Evaluation")
    print("=" * 70)
    print(f"M (negatives for scoring): {M_NEGATIVES} (fixed)")
    print(f"Aggregation method: {args.aggregation}")
    print(f"Target FPR for threshold: {args.target_fpr}")
    print("=" * 70)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model, config = load_model(args.checkpoint, device)
    action_chunk = config.get("action_chunk", 16)

    # Load episodes
    print("Loading episodes...")
    episodes = load_episodes(args.data_dir)
    print(f"Loaded {len(episodes)} episodes")

    # Split into train/val for threshold calibration
    np.random.seed(42)
    indices = np.random.permutation(len(episodes))
    n_val = len(episodes) // 5
    val_episodes = [episodes[i] for i in indices[:n_val]]
    test_episodes = [episodes[i] for i in indices[n_val:]]

    print(f"Split: {len(val_episodes)} validation, {len(test_episodes)} test")

    # Build negative pool
    print("Building negative action pool...")
    negative_pool = build_negative_pool(test_episodes, action_chunk)
    print(f"Built pool of {len(negative_pool)} action chunks")

    # ========================================================================
    # 1. Evaluate clean episodes (for threshold calibration)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Calibrating threshold on clean validation episodes")
    print("=" * 70)

    clean_results = run_episode_evaluation(
        model, val_episodes, negative_pool, device, config,
        perturbation=None, n_episodes=len(val_episodes),
        aggregation=args.aggregation
    )

    clean_scores = [r['score'] for r in clean_results['scores']]
    threshold = calibrate_threshold(clean_scores, args.target_fpr)

    print(f"\nClean episode scores:")
    print(f"  Mean: {np.mean(clean_scores):.3f}")
    print(f"  Std: {np.std(clean_scores):.3f}")
    print(f"  Min: {np.min(clean_scores):.3f}")
    print(f"  5th percentile: {np.percentile(clean_scores, 5):.3f}")
    print(f"\nCalibrated threshold (τ): {threshold:.3f}")
    print(f"  (At {args.target_fpr*100:.0f}% FPR target)")

    # ========================================================================
    # 2. Evaluate perturbed episodes
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Evaluating perturbed test episodes")
    print("=" * 70)

    perturbations = ["noise", "blur", "brightness", "contrast"]
    all_results = {'clean': clean_results}
    all_trajectories = clean_results['trajectories']

    for perturb in perturbations:
        print(f"\nEvaluating {perturb}...")
        results = run_episode_evaluation(
            model, test_episodes, negative_pool, device, config,
            perturbation=perturb, n_episodes=args.n_episodes,
            aggregation=args.aggregation
        )
        all_results[perturb] = results
        all_trajectories.extend(results['trajectories'])

    # ========================================================================
    # 3. Compute metrics
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Computing metrics")
    print("=" * 70)

    metrics = {
        'M_negatives': M_NEGATIVES,
        'aggregation': args.aggregation,
        'threshold': threshold,
        'target_fpr': args.target_fpr,
    }

    # OOD AUROC: clean vs perturbed
    print("\n--- OOD Detection (Clean vs Perturbed) ---")
    for perturb in perturbations:
        clean_ep_scores = [r['score'] for r in all_results['clean']['scores']]
        perturbed_ep_scores = [r['score'] for r in all_results[perturb]['scores']]

        # Labels: 0 = clean (negative), 1 = perturbed (positive to detect)
        scores = clean_ep_scores + perturbed_ep_scores
        labels = [0] * len(clean_ep_scores) + [1] * len(perturbed_ep_scores)

        # For AUROC, we want lower TAP score -> higher anomaly score
        anomaly_scores = [1 - s for s in scores]
        auroc = compute_auroc(anomaly_scores, labels)

        # Detection rate at threshold
        tpr, fpr = compute_detection_rates(scores, labels, threshold)

        print(f"{perturb}:")
        print(f"  OOD AUROC: {auroc:.3f}" if auroc else f"  OOD AUROC: N/A")
        print(f"  Detection rate at τ={threshold:.3f}: {tpr:.1%}")
        print(f"  False alarm rate: {fpr:.1%}")

        metrics[f'ood_auroc_{perturb}'] = auroc
        metrics[f'ood_tpr_{perturb}'] = tpr
        metrics[f'ood_fpr_{perturb}'] = fpr

    # Failure AUROC: success vs failure under perturbation
    print("\n--- Failure Detection (Success vs Failure under Perturbation) ---")
    for perturb in perturbations:
        perturbed = all_results[perturb]['scores']
        success_scores = [r['score'] for r in perturbed if not r['is_failure']]
        failure_scores = [r['score'] for r in perturbed if r['is_failure']]

        if len(success_scores) == 0 or len(failure_scores) == 0:
            print(f"{perturb}: Not enough samples")
            continue

        scores = success_scores + failure_scores
        labels = [0] * len(success_scores) + [1] * len(failure_scores)

        anomaly_scores = [1 - s for s in scores]
        auroc = compute_auroc(anomaly_scores, labels)

        tpr, _ = compute_detection_rates(scores, labels, threshold)

        print(f"{perturb}:")
        print(f"  Failure AUROC: {auroc:.3f}" if auroc else f"  Failure AUROC: N/A")
        print(f"  Failure detection rate at τ: {tpr:.1%}")
        print(f"  Success mean score: {np.mean(success_scores):.3f}")
        print(f"  Failure mean score: {np.mean(failure_scores):.3f}")

        metrics[f'failure_auroc_{perturb}'] = auroc
        metrics[f'failure_tpr_{perturb}'] = tpr

    # Early warning lead time
    print("\n--- Early Warning Lead Time ---")
    for perturb in perturbations:
        trajectories = [t for t in all_results[perturb]['trajectories'] if t['is_failure']]
        lead_times = []

        for traj in trajectories:
            lt = compute_early_warning_lead_time(
                traj['policy_scores'], threshold, traj['timesteps'], action_chunk
            )
            if lt > 0:
                lead_times.append(lt)

        if len(lead_times) > 0:
            print(f"{perturb}:")
            print(f"  Mean lead time: {np.mean(lead_times):.1f} steps")
            print(f"  Median lead time: {np.median(lead_times):.1f} steps")
            print(f"  Episodes with early warning: {len(lead_times)}/{len(trajectories)}")
            metrics[f'lead_time_mean_{perturb}'] = float(np.mean(lead_times))
            metrics[f'lead_time_episodes_{perturb}'] = len(lead_times)

    # ========================================================================
    # 4. Key claim: Policy drops more than expert under shift
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Key Claim - Policy vs Expert Score Drop")
    print("=" * 70)

    for perturb in perturbations:
        trajectories = all_results[perturb]['trajectories']
        policy_means = []
        expert_means = []

        for traj in trajectories:
            if traj['expert_scores']:
                policy_means.append(np.mean(traj['policy_scores']))
                expert_means.append(np.mean(traj['expert_scores']))

        if len(policy_means) > 0:
            gap = np.mean(expert_means) - np.mean(policy_means)
            print(f"{perturb}:")
            print(f"  Expert mean score: {np.mean(expert_means):.3f}")
            print(f"  Policy mean score: {np.mean(policy_means):.3f}")
            print(f"  Gap (expert - policy): {gap:.3f}")
            if gap > 0.1:
                print(f"  ✓ Policy actions score lower than expert under shift")
            metrics[f'expert_policy_gap_{perturb}'] = gap

    # ========================================================================
    # 5. Generate plots
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Generating plots")
    print("=" * 70)

    # Filter for perturbed trajectories for plotting
    perturbed_trajectories = [t for t in all_trajectories if t['is_perturbed']]
    plot_episode_trajectories(perturbed_trajectories, output_dir / "episode_trajectories.png")

    # ========================================================================
    # 6. Summary table
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"| Perturbation | OOD AUROC | Failure AUROC | Detection Rate | Lead Time |")
    print(f"|--------------|-----------|---------------|----------------|-----------|")
    for perturb in perturbations:
        ood = metrics.get(f'ood_auroc_{perturb}', 'N/A')
        fail = metrics.get(f'failure_auroc_{perturb}', 'N/A')
        tpr = metrics.get(f'ood_tpr_{perturb}', 'N/A')
        lt = metrics.get(f'lead_time_mean_{perturb}', 'N/A')

        ood_str = f"{ood:.3f}" if isinstance(ood, float) else ood
        fail_str = f"{fail:.3f}" if isinstance(fail, float) else fail
        tpr_str = f"{tpr:.1%}" if isinstance(tpr, float) else tpr
        lt_str = f"{lt:.1f}" if isinstance(lt, float) else lt

        print(f"| {perturb:12} | {ood_str:9} | {fail_str:13} | {tpr_str:14} | {lt_str:9} |")

    print(f"\nThreshold τ = {threshold:.3f} (calibrated at {args.target_fpr*100:.0f}% FPR on clean)")
    print(f"M = {M_NEGATIVES} negatives (fixed)")
    print(f"Aggregation: {args.aggregation}")

    # Save metrics
    with open(output_dir / "episode_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'episode_metrics.json'}")


if __name__ == "__main__":
    main()
