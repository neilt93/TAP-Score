"""
Generate score trace plot for early warning visualization.

Creates a plot showing TAP-Score over time for:
- Success episode (higher scores)
- Held-out failure episode (lower scores, drops early)

Usage:
    python plot_score_trace.py --checkpoint checkpoints_contrastive/contrastive_tap_best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib.pyplot as plt

from tap.contrastive import ContrastiveTAPScore


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


def simulate_success_action(expert_action):
    noise = np.random.randn(*expert_action.shape).astype(np.float32) * 0.05
    return expert_action + noise


def simulate_failure_heldout(expert_action):
    """Stuck policy (held-out failure type)."""
    stuck = np.tile(expert_action[0:1], (expert_action.shape[0], 1))
    return stuck.astype(np.float32)


def score_action_logmargin(model, obs, action, negative_pool, device, m_negatives=15):
    neg_indices = np.random.choice(len(negative_pool), m_negatives, replace=False)
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


def score_episode_trace(model, episode, negative_pool, device, config, failure_fn=None):
    """Get per-timestep scores for an episode."""
    obs_window = config.get("obs_window", 2)
    action_chunk = config.get("action_chunk", 16)
    images = episode['images']
    actions = episode['actions']
    ep_len = len(images)

    if ep_len < obs_window + action_chunk:
        return None, None

    scores = []
    timesteps = []

    for t in range(obs_window - 1, ep_len - action_chunk):
        start_t = t - obs_window + 1
        obs = images[start_t:t + 1].astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = perturb_observation(obs)

        expert_action = actions[t:t + action_chunk].astype(np.float32)

        if failure_fn is None:
            policy_action = simulate_success_action(expert_action)
        else:
            policy_action = failure_fn(expert_action)

        margin = score_action_logmargin(model, obs, policy_action, negative_pool, device)
        scores.append(margin)
        timesteps.append(t)

    return np.array(timesteps), np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Generate Score Trace Plot")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_contrastive/contrastive_tap_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--output", type=str, default="results/figures/score_trace.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model and data...")
    model, config = load_model(args.checkpoint, device)
    episodes = load_episodes(args.data_dir)
    action_chunk = config.get("action_chunk", 16)
    negative_pool = build_negative_pool(episodes, action_chunk)

    np.random.seed(42)

    # Find an episode long enough for good visualization
    test_ep = None
    for ep in episodes[:50]:
        if len(ep['images']) > 60:
            test_ep = ep
            break

    if test_ep is None:
        test_ep = episodes[0]

    print("Computing score traces...")

    # Success trace
    t_success, scores_success = score_episode_trace(
        model, test_ep, negative_pool, device, config, failure_fn=None
    )

    # Failure trace (stuck policy)
    t_failure, scores_failure = score_episode_trace(
        model, test_ep, negative_pool, device, config, failure_fn=simulate_failure_heldout
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t_success, scores_success, 'b-', linewidth=2, label='Success (near-expert action)', alpha=0.8)
    ax.plot(t_failure, scores_failure, 'r-', linewidth=2, label='Failure (stuck policy)', alpha=0.8)

    # Add threshold line (approximate from calibration)
    threshold = -1.96  # From 5% FPR calibration
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.5, label=f'Threshold (5% FPR)')

    # Mark prefix region (first 70%)
    prefix_end = int(len(t_success) * 0.7) + t_success[0]
    ax.axvline(x=prefix_end, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Prefix boundary (70%)')
    ax.fill_betweenx([-15, 5], t_success[0], prefix_end, alpha=0.1, color='green')

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('TAP-Score (log-margin)', fontsize=12)
    ax.set_title('TAP-Score Trace: Early Warning for Off-Manifold Actions', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-12, 4])

    # Add annotation
    ax.annotate('Early detection\n(within prefix)',
                xy=(prefix_end - 10, -8),
                fontsize=10,
                ha='center',
                color='darkgreen')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print summary stats
    print(f"\nScore Statistics:")
    print(f"  Success mean: {np.mean(scores_success):.2f}")
    print(f"  Failure mean: {np.mean(scores_failure):.2f}")
    print(f"  Separation:   {np.mean(scores_success) - np.mean(scores_failure):.2f}")


if __name__ == "__main__":
    main()
