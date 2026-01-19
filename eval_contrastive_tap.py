"""
Evaluate Contrastive TAP-Score Model

Tests the key property: expert-action control should remain high under perturbations.
This verifies the model learns observation-action compatibility, not just pixel detection.

Usage:
    python eval_contrastive_tap.py --checkpoint checkpoints_contrastive/contrastive_tap_best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

from tap.contrastive import ContrastiveTAPScore


def load_model(checkpoint_path, device):
    """Load contrastive TAP model from checkpoint."""
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

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}, val_acc: {checkpoint.get('val_acc', '?'):.3f}")
    return model, config


def load_data(data_path, obs_window, action_chunk):
    """Load data for evaluation."""
    root = zarr.open_group(str(data_path), mode='r')
    images = root['data']['img'][:]
    actions = root['data']['action'][:]
    episode_ends = root['meta']['episode_ends'][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    print(f"Loaded {len(episode_ends)} episodes, {len(images)} frames")
    return images, actions, episode_starts, episode_ends


def get_sample(images, actions, episode_starts, episode_ends, obs_window, action_chunk, idx):
    """Get a single sample (obs window, action chunk)."""
    # Pick a random episode and timestep
    ep_idx = np.random.randint(len(episode_ends))
    start = episode_starts[ep_idx]
    end = episode_ends[ep_idx]
    ep_len = end - start

    min_len = obs_window + action_chunk
    if ep_len < min_len:
        return None

    t = np.random.randint(obs_window - 1, ep_len - action_chunk)
    global_t = start + t

    # Get obs window
    start_t = global_t - obs_window + 1
    obs = images[start_t:global_t + 1]  # (T, H, W, C)
    obs = obs.astype(np.float32) / 255.0
    obs = np.transpose(obs, (0, 3, 1, 2))  # (T, C, H, W)

    # Get action chunk
    action = actions[global_t:global_t + action_chunk].astype(np.float32)

    return obs, action, ep_idx, global_t


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
        factor = 1.0 + (0.5 * severity * (1 if np.random.rand() > 0.5 else -1))
        obs = np.clip(obs * factor, 0, 1)
    elif perturbation == "cutout":
        T, C, H, W = obs.shape
        cut_size = int(min(H, W) * 0.3 * severity)
        if cut_size > 0:
            x = np.random.randint(0, W - cut_size)
            y = np.random.randint(0, H - cut_size)
            obs[:, :, y:y+cut_size, x:x+cut_size] = 0

    return obs


def build_negative_pool(images, actions, episode_starts, episode_ends, obs_window, action_chunk, n_samples=1000):
    """Build a pool of action chunks for negative sampling during scoring."""
    pool = []

    for _ in range(n_samples):
        sample = get_sample(images, actions, episode_starts, episode_ends, obs_window, action_chunk, 0)
        if sample is not None:
            _, action, _, _ = sample
            pool.append(action)

    return np.stack(pool)  # (N, H, action_dim)


def score_with_negatives(model, obs, action, negative_pool, n_negatives=15, device='cpu'):
    """
    Score an (obs, action) pair by comparing against random negatives.
    Returns softmax probability that the action is the best match.
    """
    # Sample random negatives
    n_pool = len(negative_pool)
    neg_indices = np.random.choice(n_pool, min(n_negatives, n_pool), replace=False)
    neg_actions = negative_pool[neg_indices]

    # Construct candidates: [proposal, neg1, neg2, ...]
    candidates = np.concatenate([action[np.newaxis], neg_actions], axis=0)  # (1+n_neg, H, action_dim)

    # Convert to tensors
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1, T, C, H, W)
    candidates_t = torch.from_numpy(candidates).unsqueeze(0).to(device)  # (1, 1+n_neg, H, action_dim)

    # Get logits and softmax probability
    with torch.no_grad():
        logits = model(obs_t, candidates_t)  # (1, 1+n_neg)
        probs = torch.softmax(logits, dim=-1)
        score = probs[0, 0].item()  # Probability of proposal being best

    return score


def evaluate_control(model, images, actions, episode_starts, episode_ends,
                     obs_window, action_chunk, negative_pool, device,
                     perturbation=None, n_samples=200, n_negatives=15):
    """
    Evaluate control: perturbed obs + matching expert action.

    If model learns compatibility (not pixel detection), scores should remain
    HIGH even when observations are perturbed.
    """
    scores = []

    for _ in tqdm(range(n_samples), desc=f"Control ({perturbation or 'clean'})", leave=False):
        sample = get_sample(images, actions, episode_starts, episode_ends, obs_window, action_chunk, 0)
        if sample is None:
            continue

        obs, action, _, _ = sample

        # Apply perturbation if specified
        if perturbation:
            obs = perturb_observation(obs, perturbation)

        # Score with negatives
        score = score_with_negatives(model, obs, action, negative_pool, n_negatives, device)
        scores.append(score)

    return np.array(scores)


def evaluate_mismatch(model, images, actions, episode_starts, episode_ends,
                      obs_window, action_chunk, negative_pool, device,
                      n_samples=200, n_negatives=15):
    """
    Evaluate mismatch: clean obs + WRONG action.

    Scores should be LOW (random ~1/16 = 0.0625).
    """
    scores = []

    for _ in tqdm(range(n_samples), desc="Mismatch (wrong action)", leave=False):
        sample = get_sample(images, actions, episode_starts, episode_ends, obs_window, action_chunk, 0)
        if sample is None:
            continue

        obs, _, ep_idx, global_t = sample

        # Get a mismatched action (from different episode/time)
        while True:
            sample2 = get_sample(images, actions, episode_starts, episode_ends, obs_window, action_chunk, 0)
            if sample2 is None:
                continue
            _, wrong_action, ep_idx2, global_t2 = sample2
            if ep_idx2 != ep_idx or abs(global_t2 - global_t) > action_chunk * 2:
                break

        # Score with negatives (wrong action should score low)
        score = score_with_negatives(model, obs, wrong_action, negative_pool, n_negatives, device)
        scores.append(score)

    return np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Contrastive TAP-Score")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_contrastive/contrastive_tap_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_negatives", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("Contrastive TAP-Score Evaluation")
    print("=" * 60)
    print("Key test: Expert-action control should remain HIGH under perturbations")
    print("If model learns compatibility, perturbed obs + correct action -> high score")
    print("=" * 60)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)
    obs_window = config.get("obs_window", 2)
    action_chunk = config.get("action_chunk", 16)

    # Load data
    images, actions, episode_starts, episode_ends = load_data(
        args.data_dir, obs_window, action_chunk
    )

    # Build negative pool for scoring
    print("\nBuilding negative action pool for scoring...")
    negative_pool = build_negative_pool(
        images, actions, episode_starts, episode_ends,
        obs_window, action_chunk, n_samples=1000
    )
    print(f"Built pool of {len(negative_pool)} action chunks")

    # Run evaluations
    print("\n" + "=" * 60)
    print("Running Control Evaluations")
    print("=" * 60)

    random_baseline = 1.0 / (args.n_negatives + 1)
    print(f"Random baseline: {random_baseline:.3f}")

    results = {}

    # 1. Clean observation + correct action (should be HIGH)
    print("\n1. Clean obs + correct action (expert-action):")
    clean_scores = evaluate_control(
        model, images, actions, episode_starts, episode_ends,
        obs_window, action_chunk, negative_pool, device,
        perturbation=None, n_samples=args.n_samples, n_negatives=args.n_negatives
    )
    results['expert_clean'] = {
        'mean': float(np.mean(clean_scores)),
        'std': float(np.std(clean_scores)),
    }
    print(f"   Mean score: {np.mean(clean_scores):.3f} ± {np.std(clean_scores):.3f}")
    print(f"   (Should be >> {random_baseline:.3f} random baseline)")

    # 2. Perturbed observation + correct action (should STILL be HIGH if model works)
    perturbations = ["noise", "blur", "brightness", "cutout"]

    for perturb in perturbations:
        print(f"\n2. {perturb.capitalize()} obs + correct action (expert-action control):")
        perturbed_scores = evaluate_control(
            model, images, actions, episode_starts, episode_ends,
            obs_window, action_chunk, negative_pool, device,
            perturbation=perturb, n_samples=args.n_samples, n_negatives=args.n_negatives
        )
        results[f'expert_{perturb}'] = {
            'mean': float(np.mean(perturbed_scores)),
            'std': float(np.std(perturbed_scores)),
        }
        print(f"   Mean score: {np.mean(perturbed_scores):.3f} ± {np.std(perturbed_scores):.3f}")

        # Key metric: how much does score drop?
        drop = (np.mean(clean_scores) - np.mean(perturbed_scores)) / np.mean(clean_scores) * 100
        print(f"   Drop from clean: {drop:.1f}%")
        if drop < 20:
            print(f"   ✓ Good: Score remains high under perturbation")
        elif drop < 50:
            print(f"   ~ Marginal: Some pixel dependency remains")
        else:
            print(f"   ✗ Bad: Model still detects pixel changes")

    # 3. Clean observation + WRONG action (should be LOW)
    print(f"\n3. Clean obs + WRONG action (mismatch control):")
    mismatch_scores = evaluate_mismatch(
        model, images, actions, episode_starts, episode_ends,
        obs_window, action_chunk, negative_pool, device,
        n_samples=args.n_samples, n_negatives=args.n_negatives
    )
    results['mismatch'] = {
        'mean': float(np.mean(mismatch_scores)),
        'std': float(np.std(mismatch_scores)),
    }
    print(f"   Mean score: {np.mean(mismatch_scores):.3f} ± {np.std(mismatch_scores):.3f}")
    print(f"   (Should be close to {random_baseline:.3f} random baseline)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    expert_clean = results['expert_clean']['mean']
    mismatch = results['mismatch']['mean']
    separation = expert_clean - mismatch

    print(f"Expert (correct) score:  {expert_clean:.3f}")
    print(f"Mismatch (wrong) score:  {mismatch:.3f}")
    print(f"Separation:              {separation:.3f}")
    print(f"Random baseline:         {random_baseline:.3f}")

    if separation > 0.1:
        print("\n✓ Model discriminates correct from wrong actions")
    else:
        print("\n✗ Model does NOT discriminate - possible collapse")

    # Check perturbation robustness
    avg_perturbed = np.mean([results[f'expert_{p}']['mean'] for p in perturbations])
    robustness_drop = (expert_clean - avg_perturbed) / expert_clean * 100

    print(f"\nPerturbation robustness:")
    print(f"  Clean expert score:     {expert_clean:.3f}")
    print(f"  Avg perturbed score:    {avg_perturbed:.3f}")
    print(f"  Drop:                   {robustness_drop:.1f}%")

    if robustness_drop < 15:
        print("  ✓ Excellent: Model is robust to visual perturbations")
    elif robustness_drop < 30:
        print("  ~ Good: Some robustness to perturbations")
    else:
        print("  ✗ Poor: Model still sensitive to visual changes")

    # Save results
    output_path = Path(args.checkpoint).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
