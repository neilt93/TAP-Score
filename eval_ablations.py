"""
Ablation Studies for Contrastive TAP

1. M ablation: M=7 vs M=15 negatives (does separation hold?)
2. Aggregator ablation: min vs 10th percentile episode score

Usage:
    python eval_ablations.py --checkpoint checkpoints_contrastive/contrastive_tap_best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from tap.contrastive import ContrastiveTAPScore


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
    """Add noise perturbation."""
    noise = np.random.randn(*obs.shape).astype(np.float32) * (0.3 * severity)
    return np.clip(obs + noise, 0, 1)


def simulate_policy_action(expert_action, is_failure, severity=0.3):
    """Simulate policy action."""
    if is_failure:
        noise = np.random.randn(*expert_action.shape).astype(np.float32) * severity * 2
        return expert_action + noise
    else:
        noise = np.random.randn(*expert_action.shape).astype(np.float32) * severity * 0.3
        return expert_action + noise


def score_action(model, obs, action, negative_pool, device, m_negatives):
    """Score action with specified M."""
    neg_indices = np.random.choice(len(negative_pool), m_negatives, replace=False)
    neg_actions = negative_pool[neg_indices]
    candidates = np.concatenate([action[np.newaxis], neg_actions], axis=0)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    candidates_t = torch.from_numpy(candidates).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(obs_t, candidates_t)
        probs = torch.softmax(logits, dim=-1)
        score = probs[0, 0].item()
    return score


def score_episode(model, episode, negative_pool, device, config, m_negatives,
                  perturbation=False, is_failure=False):
    """Score all timesteps in episode."""
    obs_window = config.get("obs_window", 2)
    action_chunk = config.get("action_chunk", 16)
    images = episode['images']
    actions = episode['actions']
    ep_len = len(images)

    if ep_len < obs_window + action_chunk:
        return None

    scores = []
    for t in range(obs_window - 1, ep_len - action_chunk):
        start_t = t - obs_window + 1
        obs = images[start_t:t + 1].astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))

        if perturbation:
            obs = perturb_observation(obs)

        expert_action = actions[t:t + action_chunk].astype(np.float32)
        policy_action = simulate_policy_action(expert_action, is_failure)
        score = score_action(model, obs, policy_action, negative_pool, device, m_negatives)
        scores.append(score)

    return np.array(scores)


def aggregate_episode(scores, method):
    """Aggregate timestep scores to episode score."""
    if method == 'min':
        return np.min(scores)
    elif method == 'percentile_10':
        return np.percentile(scores, 10)
    elif method == 'percentile_5':
        return np.percentile(scores, 5)
    elif method == 'mean':
        return np.mean(scores)


def run_ablation(model, episodes, negative_pool, device, config,
                 m_negatives, aggregation, n_episodes=30):
    """Run evaluation with specific M and aggregation."""
    np.random.seed(42)
    selected = np.random.choice(len(episodes), min(n_episodes, len(episodes)), replace=False)

    success_scores = []
    failure_scores = []

    for idx in tqdm(selected, desc=f"M={m_negatives}, agg={aggregation}", leave=False):
        episode = episodes[idx]

        # Success episode (with perturbation)
        scores_success = score_episode(
            model, episode, negative_pool, device, config,
            m_negatives, perturbation=True, is_failure=False
        )
        if scores_success is not None:
            success_scores.append(aggregate_episode(scores_success, aggregation))

        # Failure episode (with perturbation)
        scores_failure = score_episode(
            model, episode, negative_pool, device, config,
            m_negatives, perturbation=True, is_failure=True
        )
        if scores_failure is not None:
            failure_scores.append(aggregate_episode(scores_failure, aggregation))

    # Compute metrics
    all_scores = success_scores + failure_scores
    labels = [0] * len(success_scores) + [1] * len(failure_scores)
    anomaly_scores = [1 - s for s in all_scores]

    try:
        auroc = roc_auc_score(labels, anomaly_scores)
    except ValueError:
        auroc = None

    separation = np.mean(success_scores) - np.mean(failure_scores)

    return {
        'm_negatives': m_negatives,
        'aggregation': aggregation,
        'success_mean': float(np.mean(success_scores)),
        'failure_mean': float(np.mean(failure_scores)),
        'separation': float(separation),
        'failure_auroc': float(auroc) if auroc else None,
    }


def main():
    parser = argparse.ArgumentParser(description="TAP Ablation Studies")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_contrastive/contrastive_tap_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("Ablation Studies for Contrastive TAP")
    print("=" * 70)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    print("Loading model and data...")
    model, config = load_model(args.checkpoint, device)
    episodes = load_episodes(args.data_dir)
    action_chunk = config.get("action_chunk", 16)
    negative_pool = build_negative_pool(episodes, action_chunk)

    print(f"Loaded {len(episodes)} episodes, pool of {len(negative_pool)} negatives")

    # Ablation configurations
    m_values = [7, 15, 31]
    aggregation_methods = ['min', 'percentile_10', 'percentile_5']

    results = []

    # M ablation (fixed aggregation = percentile_10)
    print("\n" + "=" * 70)
    print("Ablation 1: Number of Negatives (M)")
    print("=" * 70)

    for m in m_values:
        result = run_ablation(
            model, episodes, negative_pool, device, config,
            m_negatives=m, aggregation='percentile_10', n_episodes=args.n_episodes
        )
        results.append(result)
        print(f"\nM = {m}:")
        print(f"  Success mean: {result['success_mean']:.3f}")
        print(f"  Failure mean: {result['failure_mean']:.3f}")
        print(f"  Separation: {result['separation']:.3f}")
        print(f"  Failure AUROC: {result['failure_auroc']:.3f}" if result['failure_auroc'] else "  Failure AUROC: N/A")

    # Aggregation ablation (fixed M = 15)
    print("\n" + "=" * 70)
    print("Ablation 2: Aggregation Method")
    print("=" * 70)

    for agg in aggregation_methods:
        result = run_ablation(
            model, episodes, negative_pool, device, config,
            m_negatives=15, aggregation=agg, n_episodes=args.n_episodes
        )
        results.append(result)
        print(f"\n{agg}:")
        print(f"  Success mean: {result['success_mean']:.3f}")
        print(f"  Failure mean: {result['failure_mean']:.3f}")
        print(f"  Separation: {result['separation']:.3f}")
        print(f"  Failure AUROC: {result['failure_auroc']:.3f}" if result['failure_auroc'] else "  Failure AUROC: N/A")

    # Summary tables
    print("\n" + "=" * 70)
    print("SUMMARY TABLES")
    print("=" * 70)

    print("\nM Ablation (aggregation = percentile_10):")
    print("| M  | Success | Failure | Separation | Failure AUROC |")
    print("|----|---------|---------|------------|---------------|")
    for r in results[:3]:
        auroc_str = f"{r['failure_auroc']:.3f}" if r['failure_auroc'] else "N/A"
        print(f"| {r['m_negatives']:2} | {r['success_mean']:.3f}   | {r['failure_mean']:.3f}   | {r['separation']:.3f}      | {auroc_str:13} |")

    print("\nAggregation Ablation (M = 15):")
    print("| Method        | Success | Failure | Separation | Failure AUROC |")
    print("|---------------|---------|---------|------------|---------------|")
    for r in results[3:]:
        auroc_str = f"{r['failure_auroc']:.3f}" if r['failure_auroc'] else "N/A"
        print(f"| {r['aggregation']:13} | {r['success_mean']:.3f}   | {r['failure_mean']:.3f}   | {r['separation']:.3f}      | {auroc_str:13} |")

    # Save results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
