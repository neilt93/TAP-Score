"""
TAP-Guided Resampling Intervention

Demonstrates practical value: at each step, sample K action candidates
and execute the one with highest TAP score.

Compares:
- Baseline: random selection from K candidates
- TAP-guided: select highest TAP score among K candidates

Usage:
    python eval_resampling_intervention.py --checkpoint checkpoints_contrastive/contrastive_tap_best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

from tap.contrastive import ContrastiveTAPScore


# Fixed parameters for consistency
M_NEGATIVES = 15
K_CANDIDATES = 8  # Number of action candidates to sample


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
        temperature=config.get("temperature", 0.1),
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
    """Build pool for TAP scoring."""
    pool = []
    for _ in range(n_samples):
        ep = episodes[np.random.randint(len(episodes))]
        if len(ep['actions']) < action_chunk:
            continue
        t = np.random.randint(0, len(ep['actions']) - action_chunk)
        pool.append(ep['actions'][t:t + action_chunk].astype(np.float32))
    return np.stack(pool)


def perturb_observation(obs, perturbation, severity=0.5):
    """Apply perturbation."""
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
    return obs


def generate_action_candidates(expert_action, k_candidates, noise_levels):
    """
    Generate K action candidates by adding different noise levels to expert.

    This simulates what a diffusion policy might produce: multiple samples
    with varying quality. One will be close to expert, others more noisy.
    """
    candidates = []

    for i in range(k_candidates):
        noise_std = noise_levels[i % len(noise_levels)]
        noise = np.random.randn(*expert_action.shape).astype(np.float32) * noise_std
        candidate = expert_action + noise
        candidates.append(candidate)

    return np.stack(candidates)  # (K, chunk_size, action_dim)


def score_action(model, obs, action, negative_pool, device, m_negatives=M_NEGATIVES):
    """Score single action with TAP."""
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


def compute_action_quality(selected_action, expert_action):
    """
    Compute how close selected action is to expert.

    Returns MSE (lower is better).
    """
    return np.mean((selected_action - expert_action) ** 2)


def run_episode_with_resampling(model, episode, negative_pool, device, config,
                                 perturbation=None, use_tap=True, k_candidates=K_CANDIDATES):
    """
    Run episode with TAP-guided or random action selection.

    Returns:
        - qualities: list of action quality scores (MSE to expert)
        - tap_scores: list of TAP scores for selected actions
    """
    obs_window = config.get("obs_window", 2)
    action_chunk = config.get("action_chunk", 16)

    images = episode['images']
    actions = episode['actions']
    ep_len = len(images)

    if ep_len < obs_window + action_chunk:
        return None, None

    # Noise levels for generating candidates (simulating diffusion samples)
    # Lower noise = closer to expert = should score higher with TAP
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    qualities = []
    tap_scores = []

    for t in range(obs_window - 1, ep_len - action_chunk, action_chunk):
        # Get observation
        start_t = t - obs_window + 1
        obs = images[start_t:t + 1].astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))

        if perturbation:
            obs = perturb_observation(obs, perturbation)

        # Get expert action
        expert_action = actions[t:t + action_chunk].astype(np.float32)

        # Generate candidates
        candidates = generate_action_candidates(expert_action, k_candidates, noise_levels)

        if use_tap:
            # Score all candidates with TAP
            scores = []
            for i in range(k_candidates):
                score = score_action(model, obs, candidates[i], negative_pool, device)
                scores.append(score)

            # Select best
            best_idx = np.argmax(scores)
            selected = candidates[best_idx]
            selected_tap = scores[best_idx]
        else:
            # Random selection
            best_idx = np.random.randint(k_candidates)
            selected = candidates[best_idx]
            selected_tap = score_action(model, obs, selected, negative_pool, device)

        # Compute quality
        quality = compute_action_quality(selected, expert_action)
        qualities.append(quality)
        tap_scores.append(selected_tap)

    return qualities, tap_scores


def main():
    parser = argparse.ArgumentParser(description="TAP-Guided Resampling Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_contrastive/contrastive_tap_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--k_candidates", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("TAP-Guided Resampling Intervention")
    print("=" * 70)
    print(f"K candidates per step: {args.k_candidates}")
    print(f"M negatives for scoring: {M_NEGATIVES}")
    print("=" * 70)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    print("\nLoading model and data...")
    model, config = load_model(args.checkpoint, device)
    episodes = load_episodes(args.data_dir)
    action_chunk = config.get("action_chunk", 16)

    print(f"Loaded {len(episodes)} episodes")

    # Build negative pool
    negative_pool = build_negative_pool(episodes, action_chunk)
    print(f"Built pool of {len(negative_pool)} action chunks")

    # Select episodes
    np.random.seed(42)
    selected_indices = np.random.choice(len(episodes), min(args.n_episodes, len(episodes)), replace=False)

    perturbations = [None, "noise", "blur", "brightness"]
    results = {}

    for perturb in perturbations:
        perturb_name = perturb or "clean"
        print(f"\n{'='*70}")
        print(f"Condition: {perturb_name}")
        print("=" * 70)

        tap_qualities = []
        random_qualities = []
        tap_scores_list = []

        for idx in tqdm(selected_indices, desc=f"Episodes ({perturb_name})"):
            episode = episodes[idx]

            # TAP-guided selection
            q_tap, s_tap = run_episode_with_resampling(
                model, episode, negative_pool, device, config,
                perturbation=perturb, use_tap=True, k_candidates=args.k_candidates
            )

            # Random selection
            q_rand, s_rand = run_episode_with_resampling(
                model, episode, negative_pool, device, config,
                perturbation=perturb, use_tap=False, k_candidates=args.k_candidates
            )

            if q_tap is not None:
                tap_qualities.extend(q_tap)
                random_qualities.extend(q_rand)
                tap_scores_list.extend(s_tap)

        # Compute statistics
        tap_mean_quality = np.mean(tap_qualities)
        rand_mean_quality = np.mean(random_qualities)
        improvement = (rand_mean_quality - tap_mean_quality) / rand_mean_quality * 100

        print(f"\nAction Quality (MSE to expert, lower is better):")
        print(f"  Random selection:     {rand_mean_quality:.4f}")
        print(f"  TAP-guided selection: {tap_mean_quality:.4f}")
        print(f"  Improvement:          {improvement:.1f}%")

        if improvement > 0:
            print(f"  ✓ TAP-guided selection improves action quality")
        else:
            print(f"  ✗ No improvement from TAP guidance")

        results[perturb_name] = {
            'random_quality': float(rand_mean_quality),
            'tap_quality': float(tap_mean_quality),
            'improvement_pct': float(improvement),
            'mean_tap_score': float(np.mean(tap_scores_list)),
            'n_steps': len(tap_qualities),
        }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Action Quality Improvement with TAP-Guided Resampling")
    print("=" * 70)
    print(f"| Condition   | Random MSE | TAP MSE   | Improvement |")
    print(f"|-------------|------------|-----------|-------------|")
    for perturb_name, r in results.items():
        print(f"| {perturb_name:11} | {r['random_quality']:.4f}     | {r['tap_quality']:.4f}    | {r['improvement_pct']:+.1f}%       |")

    print(f"\nK = {args.k_candidates} candidates, M = {M_NEGATIVES} negatives")

    # Save results
    with open(output_dir / "resampling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'resampling_results.json'}")


if __name__ == "__main__":
    main()
